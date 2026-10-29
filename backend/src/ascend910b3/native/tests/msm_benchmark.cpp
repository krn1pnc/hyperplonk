#include "msm.h"
#include "runtime_audit.h"
#include <algorithm>
#include <charconv>
#include <cstdint>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string_view>
#include <stdexcept>
#include <vector>
#include <unistd.h>
using namespace hyperplonk::ascend910b3;
using namespace hyperplonk::ascend910b3::detail;
namespace audit=hyperplonk::ascend910b3::test;
void check(aclError e,const char* label){if(e!=ACL_SUCCESS)throw std::runtime_error(std::string(label)+": "+std::to_string(e));}
void require(bool b,const char* label){if(!b)throw std::runtime_error(label);}
void load(const std::filesystem::path& path,void* dst,size_t bytes){
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    require(bool(f),"missing input");require(f.tellg()==static_cast<std::streamoff>(bytes),"wrong input length");
    f.seekg(0);f.read(static_cast<char*>(dst),bytes);require(bool(f),"truncated input");
}
struct Resources{
    aclrtStream stream{};std::vector<void*> device,host;bool initialized=false,selected=false;
    void init(){check(aclInit(nullptr),"aclInit");initialized=true;check(aclrtSetDevice(0),"setDevice");selected=true;check(aclrtCreateStream(&stream),"createStream");}
    void* d(size_t bytes){void*p{};check(aclrtMalloc(&p,bytes,ACL_MEM_MALLOC_HUGE_FIRST),"malloc");device.push_back(p);return p;}
    void* h(size_t bytes){void*p{};check(aclrtMallocHost(&p,bytes+63),"mallocHost");host.push_back(p);return reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(p)+63)&~uintptr_t(63));}
    void finish(){
        if(stream)check(aclrtSynchronizeStream(stream),"cleanupSync");
        for(void*p:device)check(aclrtFree(p),"free");device.clear();
        for(void*p:host)check(aclrtFreeHost(p),"freeHost");host.clear();
        if(stream){check(aclrtDestroyStream(stream),"destroyStream");stream=nullptr;}
        if(selected){check(aclrtResetDevice(0),"resetDevice");selected=false;}
        if(initialized){check(aclFinalize(),"finalize");initialized=false;}
    }
    ~Resources(){audit::end_runtime_audit();try{finish();}catch(const std::exception&e){std::cerr<<e.what()<<'\n';}}
};
struct Options {
    uint32_t warmups=1,samples=1,window_bits=0,count=1U<<20;
};
void usage(std::ostream& out){
    out<<"usage: msm_benchmark INPUT_DIRECTORY OUTPUT_DIRECTORY [--warmups N] [--samples N] [--window-bits 0|2..16] [--count N]\n"
        <<"Ordinary complete-MSM benchmark; profiling is not enabled.\n"
        <<"Defaults: 1 warmup, 1 target sample; warmups >= 0, samples >= 1.\n"
        <<"Loads/uploads inputs and allocates borrowed scratch once. Every call must\n"
        <<"exactly match expected.bin and pass strict allocation/copy/sync audits.\n"
        <<"Targets time run_msm entry through final 96-byte D2H and stream completion,\n"
        <<"including metadata handoffs, excluding setup and exact-point comparison.\n";
}
uint32_t parse_count(const char* text,bool nonzero){
    const std::string_view value(text);
    require(!value.empty(),"empty unsigned count");
    for(char c:value)require(c>='0'&&c<='9',"count must be unsigned decimal");
    uint32_t count=0;
    const auto parsed=std::from_chars(value.data(),value.data()+value.size(),count);
    require(parsed.ec==std::errc{}&&parsed.ptr==value.data()+value.size(),"unsigned count out of range");
    require(!nonzero||count!=0,"count must be at least 1");
    return count;
}
struct CallResult {
    double elapsed_s=0,api_s=0;
    audit::RuntimeCounts compute{},download{};
};
double median(std::vector<double>& values){
    std::sort(values.begin(),values.end());
    const size_t middle=values.size()/2;
    return values.size()%2?values[middle]:(values[middle-1]+values[middle])/2;
}
void write_counts(std::ostream& out,const audit::RuntimeCounts& c){
    out<<"{\"h2d_calls\":"<<c.h2d_calls<<",\"h2d_bytes\":"<<c.h2d_bytes
        <<",\"d2h_calls\":"<<c.d2h_calls<<",\"d2h_bytes\":"<<c.d2h_bytes
        <<",\"other_copy_calls\":"<<c.other_copy_calls<<",\"unexpected_copy_calls\":"<<c.unexpected_copy_calls
        <<",\"stream_sync_calls\":"<<c.stream_sync_calls<<",\"device_sync_calls\":"<<c.device_sync_calls
        <<",\"malloc_calls\":"<<c.malloc_calls<<",\"free_calls\":"<<c.free_calls
        <<",\"host_malloc_calls\":"<<c.host_malloc_calls<<",\"host_free_calls\":"<<c.host_free_calls<<"}";
}
int main(int argc,char**argv){try{
    if(argc==2&&std::string_view(argv[1])=="--help"){usage(std::cout);return 0;}
    if(argc<3){usage(std::cerr);throw std::runtime_error("missing input/output directories");}
    Options options;
    bool have_warmups=false,have_samples=false,have_window=false,have_count=false;
    for(int i=3;i<argc;++i){
        const std::string_view flag(argv[i]);
        require(flag=="--warmups"||flag=="--samples"||flag=="--window-bits"||flag=="--count","unknown option");
        require(i+1<argc,"missing option value");
        const bool samples=flag=="--samples",window=flag=="--window-bits",input_count=flag=="--count";
        require(input_count?!have_count:(window?!have_window:(samples?!have_samples:!have_warmups)),"duplicate option");
        const uint32_t count=parse_count(argv[++i],samples||input_count);
        if(window){
            require(count==0||(count>=2&&count<=16),"window bits must be 0 or 2..16");
            options.window_bits=count;have_window=true;
        }else if(input_count){options.count=count;have_count=true;}
        else if(samples){options.samples=count;have_samples=true;}
        else{options.warmups=count;have_warmups=true;}
    }
    const std::filesystem::path root(argv[1]),output(argv[2]);
    std::cout<<"MSM_SETUP_BEGIN warmups="<<options.warmups<<" samples="<<options.samples<<std::endl;
    std::filesystem::copy_file("/proc/self/exe",output/"loaded-executable.elf",std::filesystem::copy_options::overwrite_existing);
    std::ifstream maps("/proc/self/maps");std::ofstream(output/"process-maps.txt")<<maps.rdbuf();
    std::ofstream(output/"process-identity.txt")<<"pid="<<getpid()<<"\nexe="<<std::filesystem::read_symlink("/proc/self/exe")<<"\n";
    const size_t n=options.count;Resources r;r.init();
    MsmPlan p{};require(build_msm_plan(n,p,options.window_bits),"MSM platform unavailable");
    auto* hb=static_cast<G1Affine*>(r.h(n*96));auto* hs=static_cast<Fr*>(r.h(n*32));
    auto* result=static_cast<G1Affine*>(r.h(96));G1Affine expected{};
    std::cout<<"MSM_LOAD_INPUTS_BEGIN"<<std::endl;
    load(root/"bases.bin",hb,n*96);load(root/"scalars.bin",hs,n*32);load(root/"expected.bin",&expected,96);
    auto* b=static_cast<G1Affine*>(r.d(n*96));auto* s=static_cast<Fr*>(r.d(n*32));
    auto* out=static_cast<G1Affine*>(r.d(96));void*w=r.d(p.workspace_bytes),*h=r.h(p.host_workspace_bytes);
    // Reserve all benchmark records before any complete-MSM call.
    std::vector<CallResult> warmups(options.warmups),targets(options.samples);
    std::vector<double> elapsed_samples(options.samples),api_samples(options.samples);
    std::cout<<"MSM_METADATA_MODEL_BEGIN"<<std::endl;
    // Independent scalar-bit histogram model; outside all timed/audited compute.
    uint64_t expected_h2d=0,active=0;
    const uint64_t stride=((uint64_t(p.buckets)+8)/8)*8*8;
    for(uint32_t window=0;window<p.windows;++window){
        std::vector<uint64_t> lengths(p.buckets);
        for(size_t i=0;i<n;++i){uint32_t digit=0;
            for(uint32_t bit=0;bit<p.window_bits&&window*p.window_bits+bit<255;++bit){
                const uint32_t pos=window*p.window_bits+bit;
                digit|=((hs[i].words[pos/32]>>(pos%32))&1u)<<bit;
            }++lengths[digit];
        }
        const uint64_t largest=*std::max_element(lengths.begin()+1,lengths.end());
        if(largest){++active;uint32_t levels=0;for(uint64_t q=(largest+63)/64;q>1;q=(q+63)/64)++levels;
            uint64_t tasks=0;
            for(uint32_t b=1;b<p.buckets;++b) tasks+=(lengths[b]+63)/64;
            expected_h2d+=stride*(levels+1)+tasks*sizeof(uint32_t);}
    }
    std::cout<<"MSM_INITIAL_H2D_BEGIN"<<std::endl;
    const auto setup_start=std::chrono::steady_clock::now();
    check(aclrtMemcpyAsync(b,n*96,hb,n*96,ACL_MEMCPY_HOST_TO_DEVICE,r.stream),"upload bases");
    check(aclrtMemcpyAsync(s,n*32,hs,n*32,ACL_MEMCPY_HOST_TO_DEVICE,r.stream),"upload scalars");
    check(aclrtSynchronizeStream(r.stream),"setupSync");
    const double upload_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-setup_start).count();
    const audit::RuntimeCopyAllowance allowed[]{
        {static_cast<uint8_t*>(w)+p.region_offset[static_cast<unsigned>(MsmRegion::RowPtr)],h,size_t(8*(uint64_t(p.buckets)+1)),false},
        {static_cast<uint8_t*>(w)+p.region_offset[static_cast<unsigned>(MsmRegion::TaskPrefix)],static_cast<uint8_t*>(h)+p.host_prefix_offset,size_t(p.region_size[static_cast<unsigned>(MsmRegion::TaskPrefix)]),true}};
    std::cout<<"NPU_INPUTS_READY count="<<n<<" c="<<p.window_bits<<" windows="<<p.windows
        <<" aiv="<<p.platform.aiv_cores<<" workspace="<<p.workspace_bytes<<" host_workspace="<<p.host_workspace_bytes<<std::endl;
    const audit::RuntimeCopyAllowance final_allowed[]{ {out,result,96,false} };
    const auto run_call=[&](const char* phase,size_t index,CallResult& call){
        std::cout<<"MSM_"<<phase<<"_BEGIN index="<<index<<std::endl;
        audit::begin_runtime_audit(allowed,2);
        const auto start=std::chrono::steady_clock::now();
        const aclError status=run_msm(r.stream,b,s,out,w,p.workspace_bytes,h,p.host_workspace_bytes,p);
        const auto returned=std::chrono::steady_clock::now();
        call.compute=audit::end_runtime_audit();check(status,"run_msm");
        // Separate, exact-address allowance: no compute copy budget may cover the result.
        audit::begin_runtime_audit(final_allowed,1);
        check(aclrtMemcpyAsync(result,96,out,96,ACL_MEMCPY_DEVICE_TO_HOST,r.stream),"finalDownload");
        check(aclrtSynchronizeStream(r.stream),"finalSync");
        const auto ready=std::chrono::steady_clock::now();
        call.download=audit::end_runtime_audit();
        call.elapsed_s=std::chrono::duration<double>(ready-start).count();
        call.api_s=std::chrono::duration<double>(returned-start).count();
        std::cout<<std::fixed<<std::setprecision(9)<<"MSM_"<<phase<<"_RAW index="<<index
            <<" npu_api_to_return_s="<<call.api_s
            <<" npu_api_to_host_point_ready_s="<<call.elapsed_s<<std::endl;
        const auto& final_counts=call.download;
        require(final_counts.d2h_calls==1&&final_counts.d2h_bytes==96&&
            !final_counts.h2d_calls&&!final_counts.h2d_bytes&&final_counts.stream_sync_calls==1&&
            !final_counts.other_copy_calls&&!final_counts.unexpected_copy_calls&&!final_counts.device_sync_calls&&
            !final_counts.malloc_calls&&!final_counts.free_calls&&
            !final_counts.host_malloc_calls&&!final_counts.host_free_calls,
            "unexpected final download resources");
        const auto& compute=call.compute;
        require(compute.d2h_calls==p.windows&&compute.d2h_bytes==uint64_t(p.windows)*8*(uint64_t(p.buckets)+1),"D2H budget mismatch");
        require(compute.h2d_calls==active&&compute.h2d_bytes==expected_h2d&&compute.stream_sync_calls==p.windows,"H2D/sync budget mismatch");
        require(!compute.other_copy_calls&&!compute.unexpected_copy_calls&&!compute.device_sync_calls&&!compute.malloc_calls&&!compute.free_calls,"unexpected compute resources");
        // Unlike the diagnostic profile executable, ordinary calls allow no profiler allocations.
        require(!compute.host_malloc_calls&&!compute.host_free_calls,"unexpected compute host allocations");
        require(std::memcmp(result,&expected,96)==0,"CPU/NPU point mismatch");
        std::cout<<"MSM_"<<phase<<"_PASS index="<<index<<" exact_match=true strict_resource_audit=true"<<std::endl;
    };
    for(size_t i=0;i<warmups.size();++i)run_call("WARMUP",i+1,warmups[i]);
    for(size_t i=0;i<targets.size();++i){
        run_call("TARGET",i+1,targets[i]);
        elapsed_samples[i]=targets[i].elapsed_s;api_samples[i]=targets[i].api_s;
    }
    const double median_s=median(elapsed_samples),median_api_s=median(api_samples);
    std::cout<<"MSM_REPORT_BEGIN"<<std::endl;
    std::ostringstream hex;hex<<std::hex<<std::setfill('0');for(size_t i=0;i<96;++i)hex<<std::setw(2)<<unsigned(reinterpret_cast<uint8_t*>(result)[i]);
    const auto& compute=targets.front().compute;
    std::ostringstream report;report<<std::fixed<<std::setprecision(9)
        <<"{\"count\":"<<n<<",\"samples\":"<<options.samples<<",\"warmup\":"<<options.warmups<<",\"window_bits\":"<<p.window_bits<<",\"windows\":"<<p.windows
        <<",\"aiv_cores\":"<<p.platform.aiv_cores<<",\"device_workspace_bytes\":"<<p.workspace_bytes<<",\"host_workspace_bytes\":"<<p.host_workspace_bytes
        <<",\"initial_h2d_s\":"<<upload_s<<",\"npu_api_to_return_s\":"<<median_api_s<<",\"npu_api_to_host_point_ready_s\":"<<median_s
        <<",\"median_s\":"<<median_s<<",\"timing_scope\":\"run_msm_entry_through_final_96_byte_D2H_and_stream_completion\""
        <<",\"metadata_d2h_bytes\":"<<compute.d2h_bytes<<",\"metadata_h2d_bytes\":"<<compute.h2d_bytes<<",\"metadata_sync\":"<<compute.stream_sync_calls
        <<",\"unexpected_copy\":"<<compute.unexpected_copy_calls<<",\"compute_device_alloc\":"<<compute.malloc_calls<<",\"compute_device_free\":"<<compute.free_calls
        <<",\"compute_host_malloc_calls\":"<<compute.host_malloc_calls<<",\"compute_host_free_calls\":"<<compute.host_free_calls
        <<",\"samples_s\":[";
    for(size_t i=0;i<targets.size();++i){if(i)report<<',';report<<targets[i].elapsed_s;}
    report<<"],\"warmups_s\":[";
    for(size_t i=0;i<warmups.size();++i){if(i)report<<',';report<<warmups[i].elapsed_s;}
    report<<"],\"calls\":[";
    bool first=true;
    const auto write_calls=[&](const char* phase,const std::vector<CallResult>& calls){
        for(size_t i=0;i<calls.size();++i){
            if(!first)report<<',';first=false;
            report<<"{\"phase\":\""<<phase<<"\",\"index\":"<<i+1
                <<",\"npu_api_to_return_s\":"<<calls[i].api_s
                <<",\"npu_api_to_host_point_ready_s\":"<<calls[i].elapsed_s<<",\"compute\":";
            write_counts(report,calls[i].compute);report<<",\"final_download\":";
            write_counts(report,calls[i].download);report<<",\"exact_match\":true}";
        }
    };
    write_calls("warmup",warmups);write_calls("target",targets);
    report<<"],\"diagnostic_only\":false,\"exact_match\":true,\"result_hex\":\""<<hex.str()<<"\"}\n";
    std::ofstream report_file(output/"run-result.json");report_file<<report.str();report_file.close();
    require(bool(report_file),"failed to write benchmark report");
    std::cout<<report.str()<<"MSM_TARGET_MEDIAN seconds="<<median_s<<" samples="<<options.samples
        <<" warmups="<<options.warmups<<"\nMSM_CPU_NPU_MATCH count="<<n<<std::endl;
    r.finish();return 0;
}catch(const std::exception&e){std::cerr<<"MSM_BENCH_FAIL: "<<e.what()<<'\n';return 1;}}
