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
    struct Allocation{void* pointer;size_t bytes;};
    aclrtStream stream{};std::vector<Allocation> device,host;
    bool initialized=false,selected=false;
    uint64_t device_bytes=0,peak_device_bytes=0,host_bytes=0,peak_host_bytes=0;
    double allocation_s=0;
    Resources()=default;
    Resources(const Resources&)=delete;Resources& operator=(const Resources&)=delete;
    Resources(Resources&&)=delete;Resources& operator=(Resources&&)=delete;
    void init(){check(aclInit(nullptr),"aclInit");initialized=true;check(aclrtSetDevice(0),"setDevice");selected=true;check(aclrtCreateStream(&stream),"createStream");}
    void* d(size_t bytes){
        if(!bytes)return nullptr;
        device.push_back({nullptr,bytes});auto& allocation=device.back();
        const auto start=std::chrono::steady_clock::now();
        check(aclrtMalloc(&allocation.pointer,bytes,ACL_MEM_MALLOC_HUGE_FIRST),"malloc");
        allocation_s+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        device_bytes+=bytes;peak_device_bytes=std::max(peak_device_bytes,device_bytes);
        return allocation.pointer;
    }
    void* h(size_t bytes){
        if(!bytes)return nullptr;
        require(bytes<=SIZE_MAX-63,"host allocation size overflow");
        host.push_back({nullptr,bytes+63});auto& allocation=host.back();
        const auto start=std::chrono::steady_clock::now();
        check(aclrtMallocHost(&allocation.pointer,bytes+63),"mallocHost");
        allocation_s+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        host_bytes+=bytes+63;peak_host_bytes=std::max(peak_host_bytes,host_bytes);
        return reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(allocation.pointer)+63)&~uintptr_t(63));
    }
    void retire(void* pointer){
        check(aclrtSynchronizeStream(stream),"retireSync");
        for(auto& allocation:device)if(allocation.pointer==pointer){
            check(aclrtFree(pointer),"retireFree");allocation.pointer=nullptr;device_bytes-=allocation.bytes;return;
        }
        throw std::runtime_error("retiring unowned device allocation");
    }
    void finish(){
        // A failed drain throws before any free, stream destruction or runtime teardown.
        if(stream)check(aclrtSynchronizeStream(stream),"cleanupSync");
        for(auto& allocation:device)if(allocation.pointer){
            check(aclrtFree(allocation.pointer),"free");allocation.pointer=nullptr;device_bytes-=allocation.bytes;
        }
        device.clear();
        for(auto& allocation:host)if(allocation.pointer){
            check(aclrtFreeHost(allocation.pointer),"freeHost");allocation.pointer=nullptr;host_bytes-=allocation.bytes;
        }
        host.clear();
        if(stream){check(aclrtDestroyStream(stream),"destroyStream");stream=nullptr;}
        if(selected){check(aclrtResetDevice(0),"resetDevice");selected=false;}
        if(initialized){check(aclFinalize(),"finalize");initialized=false;}
    }
    ~Resources(){audit::end_runtime_audit();try{finish();}catch(const std::exception&e){std::cerr<<"resources retained after cleanup failure: "<<e.what()<<'\n';}}
};
struct Options {
    uint32_t warmups=1,samples=1,window_bits=0,count=1U<<20,reuse_count=3;
    bool prepared=false;
    std::filesystem::path scalar_cases;
};
void usage(std::ostream& out){
    out<<"usage: msm_benchmark INPUT_DIRECTORY OUTPUT_DIRECTORY [--mode one-shot|prepared] [--warmups N] [--samples N] [--window-bits 0|2..20] [--count N] [--scalar-cases MANIFEST.tsv] [--reuse-count R]\n"
        <<"Ordinary complete-MSM benchmark; profiling is not enabled.\n"
        <<"Default one-shot: 1 warmup, 1 sample; prepared: 1 warmup, 3 samples, reuse-count 3.\n"
        <<"Every result must exactly match its independent expected point and pass unchanged strict audits.\n"
        <<"One-shot times run_msm entry through final 96-byte D2H and stream completion.\n"
        <<"Prepared times one completed cold preparation, then R actual calls before steady warmups/samples.\n"
        <<"TSV rows: case_id<TAB>scalar_file<TAB>expected_file (optional exact header).\n"
        <<"Relative paths resolve under the manifest directory; all rows share INPUT_DIRECTORY/bases.bin.\n"
        <<"Without a manifest, reuse repeats the original scalar vector and is not varying-scalar evidence.\n";
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
    std::string case_id,result_hex;
    std::chrono::steady_clock::time_point ready{};
};
double median(std::vector<double>& values);
void write_counts(std::ostream& out,const audit::RuntimeCounts& c);
std::string json_string(const std::string& text){
    std::ostringstream out;out<<'"';
    for(unsigned char c:text){
        if(c=='"'||c=='\\')out<<'\\'<<char(c);
        else if(c<32)out<<"\\u00"<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(c)<<std::dec;
        else out<<char(c);
    }
    out<<'"';return out.str();
}
std::string point_hex(const G1Affine& point){
    std::ostringstream out;out<<std::hex<<std::setfill('0');
    for(size_t i=0;i<96;++i)out<<std::setw(2)<<unsigned(reinterpret_cast<const uint8_t*>(&point)[i]);
    return out.str();
}
struct ScalarCase{
    std::string id;std::filesystem::path scalar_file,expected_file;
    Fr* host=nullptr;Fr* device=nullptr;G1Affine expected{};
    double upload_s=0;uint64_t active=0,h2d_bytes=0;
};
std::vector<ScalarCase> scalar_cases(const Options& options,const std::filesystem::path& root){
    std::vector<ScalarCase> cases;
    // The original opaque vector remains the steady-state comparison target.
    cases.push_back({"original",root/"scalars.bin",root/"expected.bin"});
    if(options.scalar_cases.empty())return cases;
    std::ifstream file(options.scalar_cases);require(bool(file),"cannot open scalar-case manifest");
    std::string line;size_t number=0;
    while(std::getline(file,line)){
        ++number;if(!line.empty()&&line.back()=='\r')line.pop_back();
        if(number==1&&line=="case_id\tscalar_file\texpected_file")continue;
        const auto fail=[&](const char* reason){
            throw std::runtime_error(options.scalar_cases.string()+":"+std::to_string(number)+": "+reason);
        };
        const auto first=line.find('\t'),second=first==std::string::npos?first:line.find('\t',first+1);
        if(first==std::string::npos||second==std::string::npos||line.find('\t',second+1)!=std::string::npos)
            fail("expected exactly three TSV columns");
        const std::string id=line.substr(0,first),scalar=line.substr(first+1,second-first-1),expected=line.substr(second+1);
        if(id.empty()||scalar.empty()||expected.empty())fail("empty TSV column");
        for(unsigned char c:line)if((c<32&&c!='\t')||c==127)fail("control character in TSV");
        for(size_t i=1;i<cases.size();++i)if(cases[i].id==id)fail("duplicate case_id");
        const auto resolve=[&](const std::string& path){
            const std::filesystem::path p(path);return p.is_absolute()?p:options.scalar_cases.parent_path()/p;
        };
        cases.push_back({id,resolve(scalar),resolve(expected)});
    }
    require(file.eof(),"failed reading scalar-case manifest");
    require(cases.size()>1,"scalar-case manifest has no records");
    require(options.reuse_count>=cases.size()-1,"reuse-count must cover every manifest record at least once");
    return cases;
}
void model_metadata(ScalarCase& input,const MsmPlan& p,size_t n){
    const uint64_t stride=((uint64_t(p.buckets)+8)/8)*8*8;
    for(uint32_t window=0;window<p.windows;++window){
        std::vector<uint64_t> lengths(p.buckets);
        for(size_t i=0;i<n;++i){
            uint32_t digit=0;
            for(uint32_t bit=0;bit<p.window_bits&&window*p.window_bits+bit<255;++bit){
                const uint32_t pos=window*p.window_bits+bit;
                digit|=((input.host[i].words[pos/32]>>(pos%32))&1u)<<bit;
            }
            ++lengths[digit];
        }
        const uint64_t largest=*std::max_element(lengths.begin()+1,lengths.end());
        if(largest){
            ++input.active;uint32_t levels=0;
            for(uint64_t q=(largest+63)/64;q>1;q=(q+63)/64)++levels;
            uint64_t tasks=0;for(uint32_t b=1;b<p.buckets;++b)tasks+=(lengths[b]+63)/64;
            input.h2d_bytes+=stride*(levels+1)+tasks*sizeof(uint32_t);
        }
    }
}
int run_prepared_benchmark(const Options& options,const std::filesystem::path& root,const std::filesystem::path& output){
    using Clock=std::chrono::steady_clock;
    const auto seconds=[](Clock::time_point start,Clock::time_point end){return std::chrono::duration<double>(end-start).count();};
    const auto setup_start=Clock::now();
    Resources r;r.init();
    size_t free_before=0,total_memory=0;
    check(aclrtGetMemInfo(ACL_HBM_MEM,&free_before,&total_memory),"getMemInfo");
    const size_t n=options.count;
    require(n<=SIZE_MAX/96,"input byte count overflow");
    MsmPlan p{};require(build_msm_plan(n,p,options.window_bits),"MSM platform unavailable");
    MsmPreparedSizes sizes{};require(query_msm_prepared_sizes(p,sizes),"prepared size query failed");
    require(sizes.reusable_bytes<=SIZE_MAX-128&&sizes.scratch_bytes<=SIZE_MAX&&sizes.host_workspace_bytes<=SIZE_MAX,
        "prepared allocation size overflow");
    auto cases=scalar_cases(options,root);
    const size_t sequence_begin=options.scalar_cases.empty()?0:1,sequence_size=cases.size()-sequence_begin;
    require(cases.size()<=UINT64_MAX/(n*32),"scalar pool size overflow");
    const uint64_t scalar_pool_bytes=uint64_t(cases.size())*n*32;
    auto* hb=static_cast<G1Affine*>(r.h(n*96));
    auto* result=static_cast<G1Affine*>(r.h(96));
    constexpr size_t validation_chunk=1U<<20;
    auto* validation=static_cast<uint8_t*>(r.h(validation_chunk));
    std::vector<uint8_t> snapshot_chunk(validation_chunk);
    double input_load_s=0,metadata_model_s=0;
    const auto base_load_start=Clock::now();
    load(root/"bases.bin",hb,n*96);
    input_load_s+=seconds(base_load_start,Clock::now());
    for(auto& input:cases){
        input.host=static_cast<Fr*>(r.h(n*32));
        const auto load_start=Clock::now();
        try{load(input.scalar_file,input.host,n*32);load(input.expected_file,&input.expected,96);}
        catch(const std::exception& e){throw std::runtime_error("case "+input.id+": "+e.what()+" ["+input.scalar_file.string()+", "+input.expected_file.string()+"]");}
        input_load_s+=seconds(load_start,Clock::now());
        const auto model_start=Clock::now();model_metadata(input,p,n);
        metadata_model_s+=seconds(model_start,Clock::now());
    }
    bool varying=false;
    for(size_t i=sequence_begin+1;i<cases.size();++i)
        varying|=std::memcmp(cases[sequence_begin].host,cases[i].host,n*32)!=0;
    auto* bases=static_cast<G1Affine*>(r.d(n*96));
    auto* cache_allocation=static_cast<uint8_t*>(r.d(size_t(sizes.reusable_bytes)+128));
    void* cache=cache_allocation+64;
    auto* out=static_cast<G1Affine*>(r.d(96));
    void* scratch=r.d(size_t(sizes.scratch_bytes));
    void* pinned=r.h(size_t(sizes.host_workspace_bytes));
    for(auto& input:cases)input.device=static_cast<Fr*>(r.d(n*32));
    std::vector<CallResult> reuse(options.reuse_count),warmups(options.warmups),targets(options.samples);
    std::vector<double> samples(options.samples),api_samples(options.samples);
    const auto upload_start=Clock::now();
    check(aclrtMemcpyAsync(bases,n*96,hb,n*96,ACL_MEMCPY_HOST_TO_DEVICE,r.stream),"upload bases");
    check(aclrtSynchronizeStream(r.stream),"base upload sync");
    const double bases_upload_s=seconds(upload_start,Clock::now());
    for(auto& input:cases){
        const auto start=Clock::now();
        check(aclrtMemcpyAsync(input.device,n*32,input.host,n*32,ACL_MEMCPY_HOST_TO_DEVICE,r.stream),"upload scalar pool");
        check(aclrtSynchronizeStream(r.stream),"scalar upload sync");
        input.upload_s=seconds(start,Clock::now());
    }
    const auto guard_start=Clock::now();
    std::memset(validation,0xa7,64);std::memset(validation+64,0x5d,64);
    check(aclrtMemcpyAsync(cache_allocation,64,validation,64,ACL_MEMCPY_HOST_TO_DEVICE,r.stream),"cache prefix guard upload");
    check(aclrtMemcpyAsync(cache_allocation+64+sizes.reusable_bytes,64,validation+64,64,ACL_MEMCPY_HOST_TO_DEVICE,r.stream),"cache suffix guard upload");
    check(aclrtSynchronizeStream(r.stream),"cache guard upload sync");
    const double guard_upload_s=seconds(guard_start,Clock::now());
    const audit::RuntimeCopyAllowance allowed[]{
        {static_cast<uint8_t*>(scratch)+p.region_offset[static_cast<unsigned>(MsmRegion::RowPtr)],pinned,size_t(8*(uint64_t(p.buckets)+1)),false},
        {static_cast<uint8_t*>(scratch)+p.region_offset[static_cast<unsigned>(MsmRegion::TaskPrefix)],static_cast<uint8_t*>(pinned)+p.host_prefix_offset,size_t(p.region_size[static_cast<unsigned>(MsmRegion::TaskPrefix)]),true}};
    const audit::RuntimeCopyAllowance final_allowed[]{{out,result,96,false}};
    std::optional<PreparedSrsView> prepared;
    const auto run_call=[&](const char* phase,size_t index,ScalarCase& input,CallResult& call){
        call.case_id=input.id;
        std::cout<<"MSM_PREPARED_"<<phase<<"_BEGIN index="<<index<<" case_id="<<input.id<<std::endl;
        audit::begin_runtime_audit(allowed,2);
        const auto start=Clock::now();
        // No canonical-base pointer is accepted by this online API.
        const aclError status=run_msm_prepared(r.stream,*prepared,input.device,n,out,scratch,
            size_t(sizes.scratch_bytes),pinned,size_t(sizes.host_workspace_bytes),p);
        const auto returned=Clock::now();
        call.compute=audit::end_runtime_audit();check(status,"run_msm_prepared");
        audit::begin_runtime_audit(final_allowed,1);
        check(aclrtMemcpyAsync(result,96,out,96,ACL_MEMCPY_DEVICE_TO_HOST,r.stream),"finalDownload");
        check(aclrtSynchronizeStream(r.stream),"finalSync");
        call.ready=Clock::now();call.download=audit::end_runtime_audit();
        call.elapsed_s=seconds(start,call.ready);call.api_s=seconds(start,returned);
        const auto& final=call.download;
        require(final.d2h_calls==1&&final.d2h_bytes==96&&!final.h2d_calls&&!final.h2d_bytes&&
            final.stream_sync_calls==1&&!final.other_copy_calls&&!final.unexpected_copy_calls&&
            !final.device_sync_calls&&!final.malloc_calls&&!final.free_calls&&!final.host_malloc_calls&&!final.host_free_calls,
            "unexpected final download resources");
        const auto& compute=call.compute;
        require(compute.d2h_calls==p.windows&&compute.d2h_bytes==uint64_t(p.windows)*8*(uint64_t(p.buckets)+1),"D2H budget mismatch");
        require(compute.h2d_calls==input.active&&compute.h2d_bytes==input.h2d_bytes&&compute.stream_sync_calls==p.windows,"H2D/sync budget mismatch");
        require(!compute.other_copy_calls&&!compute.unexpected_copy_calls&&!compute.device_sync_calls&&
            !compute.malloc_calls&&!compute.free_calls&&!compute.host_malloc_calls&&!compute.host_free_calls,"unexpected compute resources");
        require(std::memcmp(result,&input.expected,96)==0,"CPU/NPU point mismatch");
        call.result_hex=point_hex(*result);
        std::cout<<std::fixed<<std::setprecision(9)<<"MSM_PREPARED_"<<phase<<"_PASS index="<<index
            <<" case_id="<<input.id<<" online_s="<<call.elapsed_s<<" exact_match=true strict_resource_audit=true"<<std::endl;
    };
    audit::begin_runtime_audit();
    const auto prepare_start=Clock::now();
    const aclError prepare_status=prepare_msm_srs(r.stream,bases,n*96,cache,size_t(sizes.reusable_bytes),p,prepared);
    const auto prepare_end=Clock::now();
    const auto cold_counts=audit::end_runtime_audit();
    check(prepare_status,"prepare_msm_srs");require(prepared.has_value(),"completed prepare did not publish view");
    const double cold_s=seconds(prepare_start,prepare_end);
    require(cold_counts.stream_sync_calls==1&&!cold_counts.h2d_calls&&!cold_counts.d2h_calls&&
        !cold_counts.h2d_bytes&&!cold_counts.d2h_bytes&&!cold_counts.other_copy_calls&&!cold_counts.unexpected_copy_calls&&
        !cold_counts.device_sync_calls&&!cold_counts.malloc_calls&&!cold_counts.free_calls&&
        !cold_counts.host_malloc_calls&&!cold_counts.host_free_calls,"unexpected cold preparation resources");
    const auto retirement_start=Clock::now();r.retire(bases);bases=nullptr;
    const double bases_retirement_s=seconds(retirement_start,Clock::now());
    const uint64_t resident_device_bytes=r.device_bytes;
    // Full snapshots and guard reads are validation traffic, never online/audited compute.
    // A temporary disk snapshot permits exact full-cache comparison without a second HBM-sized host allocation.
    const auto snapshot_path=output/"prepared-cache-validation.tmp";
    struct SnapshotCleanup{
        std::filesystem::path path;
        ~SnapshotCleanup(){std::error_code error;std::filesystem::remove(path,error);}
    } snapshot_cleanup{snapshot_path};
    uint64_t validation_d2h_bytes=0;
    const auto snapshot=[&](bool compare){
        std::fstream file;
        file.open(snapshot_path,std::ios::binary|(compare?std::ios::in:(std::ios::out|std::ios::trunc)));
        require(bool(file),"cannot open cache validation snapshot");
        uint64_t hash=14695981039346656037ULL;
        const size_t total=size_t(sizes.reusable_bytes)+128;
        for(size_t offset=0;offset<total;){
            const size_t bytes=std::min(validation_chunk,total-offset);
            check(aclrtMemcpyAsync(validation,validation_chunk,cache_allocation+offset,bytes,ACL_MEMCPY_DEVICE_TO_HOST,r.stream),"validation cache download");
            check(aclrtSynchronizeStream(r.stream),"validation cache sync");
            validation_d2h_bytes+=bytes;
            for(size_t j=0;j<bytes;++j){
                const size_t position=offset+j;
                if(position<64)require(validation[j]==0xa7,"cache prefix guard changed");
                else if(position>=64+sizes.reusable_bytes)require(validation[j]==0x5d,"cache suffix guard changed");
                else{hash^=validation[j];hash*=1099511628211ULL;}
            }
            if(compare){
                file.read(reinterpret_cast<char*>(snapshot_chunk.data()),bytes);require(bool(file),"truncated cache validation snapshot");
                require(std::memcmp(validation,snapshot_chunk.data(),bytes)==0,"immutable prepared cache changed");
            }else{
                file.write(reinterpret_cast<const char*>(validation),bytes);require(bool(file),"cache validation snapshot write failed");
            }
            offset+=bytes;
        }
        file.close();require(bool(file),"cache validation snapshot close failed");return hash;
    };
    const auto validation_before_start=Clock::now();
    const uint64_t hash_before=snapshot(false);
    const double validation_before_s=seconds(validation_before_start,Clock::now());
    for(size_t i=0;i<reuse.size();++i)run_call("REUSE",i+1,cases[sequence_begin+i%sequence_size],reuse[i]);
    const double reuse_wall_s=seconds(prepare_start,reuse.back().ready);
    const double setup_wall_s=seconds(setup_start,reuse.back().ready);
    double online_total=0;for(const auto& call:reuse)online_total+=call.elapsed_s;
    const double resident_total=cold_s+online_total;
    double scalar_upload_total=0;for(const auto& input:cases)scalar_upload_total+=input.upload_s;
    // Historical steady comparisons always use original scalars, independently of the cold manifest sequence.
    for(size_t i=0;i<warmups.size();++i)run_call("WARMUP",i+1,cases.front(),warmups[i]);
    for(size_t i=0;i<targets.size();++i){
        run_call("TARGET",i+1,cases.front(),targets[i]);samples[i]=targets[i].elapsed_s;api_samples[i]=targets[i].api_s;
    }
    const auto validation_after_start=Clock::now();
    const uint64_t hash_after=snapshot(true);
    const double validation_after_s=seconds(validation_after_start,Clock::now());
    require(hash_before==hash_after,"cache hash changed");
    require(std::filesystem::remove(snapshot_path),"cache validation snapshot removal failed");
    const double steady_s=median(samples),api_s=median(api_samples);
    prepared.reset();
    const auto cache_retirement_start=Clock::now();r.retire(cache_allocation);
    const double cache_retirement_s=seconds(cache_retirement_start,Clock::now());
    r.finish();
    std::ostringstream report;report<<std::fixed<<std::setprecision(9)
        <<"{\"execution_mode\":\"prepared\",\"count\":"<<n<<",\"window_bits\":"<<p.window_bits
        <<",\"windows\":"<<p.windows<<",\"aiv_cores\":"<<p.platform.aiv_cores
        <<",\"samples\":"<<options.samples<<",\"warmup\":"<<options.warmups
        <<",\"prepared_timing_scope\":\"run_msm_prepared_entry_through_final_96_byte_D2H_and_stream_completion\""
        <<",\"cold_prepare_s\":"<<cold_s<<",\"first_online_s\":"<<reuse.front().elapsed_s
        <<",\"prepared_online_median_s\":"<<steady_s<<",\"prepared_api_to_return_s\":"<<api_s
        <<",\"reuse_count\":"<<options.reuse_count<<",\"resident_compute_total_s\":"<<resident_total
        <<",\"reuse_online_total_s\":"<<online_total<<",\"reuse_amortized_resident_compute_s\":"<<resident_total/options.reuse_count
        <<",\"reuse_workload_wall_s\":"<<reuse_wall_s<<",\"setup_inclusive_wall_s\":"<<setup_wall_s
        <<",\"reuse_workload_wall_scope\":\"prepare_entry_through_Rth_host_ready_includes_base_retirement_pre_reuse_full_cache_D2H_disk_snapshot_validation_and_host_gaps_excludes_preprepare_scalar_pool_setup_uploads_and_poststeady_validation_not_HyperPlonk_proof_or_economic_preparation_cost\""
        <<",\"setup_inclusive_scope\":\"runtime_init_through_Rth_host_ready_including_file_loads_metadata_model_allocation_upload_and_pre_reuse_cache_validation\""
        <<",\"bases_file\":"<<json_string((root/"bases.bin").string())
        <<",\"oracle_policy\":\"provided_exact_96B_expected_files_independent_oracle_provenance_parent_owned\""
        <<",\"initial_bases_h2d_s\":"<<bases_upload_s<<",\"allocation_s\":"<<r.allocation_s
        <<",\"scalar_upload_total_s\":"<<scalar_upload_total<<",\"input_load_s\":"<<input_load_s
        <<",\"cache_validation_s\":"<<validation_before_s+validation_after_s
        <<",\"cache_validation_d2h_bytes\":"<<validation_d2h_bytes
        <<",\"cache_validation_disk_io_bytes\":"<<2*(sizes.reusable_bytes+128)
        <<",\"metadata_model_s\":"<<metadata_model_s<<",\"setup_before_prepare_s\":"<<seconds(setup_start,prepare_start)
        <<",\"cache_guard_upload_s\":"<<guard_upload_s<<",\"bases_retirement_s\":"<<bases_retirement_s
        <<",\"cache_retirement_s\":"<<cache_retirement_s<<",\"cache_retired_after_all_readers\":true"
        <<",\"reusable_bytes\":"<<sizes.reusable_bytes<<",\"cache_guard_bytes\":128,\"scratch_bytes\":"<<sizes.scratch_bytes
        <<",\"host_workspace_bytes\":"<<sizes.host_workspace_bytes<<",\"scalar_pool_bytes\":"<<scalar_pool_bytes
        <<",\"scalar_pool_count\":"<<cases.size()<<",\"scalar_residency_policy\":\"all_sets_uploaded_once_before_cold_prepare\""
        <<",\"peak_device_bytes\":"<<r.peak_device_bytes<<",\"resident_device_bytes_after_base_retirement\":"<<resident_device_bytes
        <<",\"peak_pinned_allocation_bytes\":"<<r.peak_host_bytes<<",\"device_memory_free_before_allocation\":"<<free_before
        <<",\"device_memory_total\":"<<total_memory<<",\"memory_accounting_scope\":\"benchmark_owned_allocations_excludes_runtime_overhead\""
        <<",\"canonical_bases_retired_before_online\":true,\"varying_scalars\":"<<(varying?"true":"false")
        <<",\"scalar_manifest\":"<<json_string(options.scalar_cases.string())
        <<",\"steady_case_id\":\"original\",\"cold_prepare_calls\":1,\"cold_prepare_audit\":";
    write_counts(report,cold_counts);
    report<<",\"kernel_launch_counts_measured\":false,\"kernel_launch_trace_gate\":\"parent_pending\""
        <<",\"cache_validation\":{\"payload_bytes\":"<<sizes.reusable_bytes<<",\"full_byte_comparison\":true"
        <<",\"guards_intact\":true,\"hash_algorithm\":\"fnv1a64_noncryptographic\",\"hash_before\":"<<json_string(std::to_string(hash_before))
        <<",\"hash_after\":"<<json_string(std::to_string(hash_after))<<",\"before_s\":"<<validation_before_s
        <<",\"after_s\":"<<validation_after_s<<",\"validation_d2h_bytes\":"<<validation_d2h_bytes
        <<",\"traffic_scope\":\"outside_online_audits_and_timing\",\"temporary_snapshot_removed\":true}"
        <<",\"scalar_upload_times_s\":[";
    for(size_t i=0;i<cases.size();++i){if(i)report<<',';report<<cases[i].upload_s;}
    report<<"],\"scalar_cases\":[";
    for(size_t i=0;i<cases.size();++i){
        if(i)report<<',';
        report<<"{\"case_id\":"<<json_string(cases[i].id)<<",\"scalar_file\":"<<json_string(cases[i].scalar_file.string())
            <<",\"expected_file\":"<<json_string(cases[i].expected_file.string())<<",\"expected_hex\":"<<json_string(point_hex(cases[i].expected))
            <<",\"scalar_bytes\":"<<n*32<<",\"expected_bytes\":96,\"upload_s\":"<<cases[i].upload_s<<'}';
    }
    report<<"],\"prepared_online_samples_s\":[";
    for(size_t i=0;i<targets.size();++i){if(i)report<<',';report<<targets[i].elapsed_s;}
    report<<"],\"prepared_warmups_s\":[";
    for(size_t i=0;i<warmups.size();++i){if(i)report<<',';report<<warmups[i].elapsed_s;}
    const auto write_calls=[&](const std::vector<CallResult>& calls){
        for(size_t i=0;i<calls.size();++i){
            if(i)report<<',';
            report<<"{\"index\":"<<i+1<<",\"case_id\":"<<json_string(calls[i].case_id)
                <<",\"online_s\":"<<calls[i].elapsed_s<<",\"api_to_return_s\":"<<calls[i].api_s<<",\"compute\":";
            write_counts(report,calls[i].compute);report<<",\"final_download\":";
            write_counts(report,calls[i].download);
            report<<",\"exact_match\":true,\"result_hex\":"<<json_string(calls[i].result_hex)<<'}';
        }
    };
    report<<"],\"reuse_calls\":[";write_calls(reuse);
    report<<"],\"warmup_calls\":[";write_calls(warmups);
    report<<"],\"steady_calls\":[";write_calls(targets);
    report<<"],\"reuse_projections\":[";
    bool first_projection=true;
    for(uint32_t count:{1u,2u,4u,8u,16u}){
        if(!first_projection)report<<',';first_projection=false;
        double projected=cold_s;for(uint32_t i=0;i<count;++i)projected+=reuse[i%reuse.size()].elapsed_s;
        report<<"{\"reuse_count\":"<<count<<",\"model_only\":true,\"projection\":\"cold_plus_cyclic_measured_online_latencies\""
            <<",\"resident_compute_total_s\":"<<projected<<",\"amortized_s\":"<<projected/count<<'}';
    }
    report<<"],\"cpu_comparison_policy\":\"NPU-prepared_vs_arkworks0.6_affine-reused_16threads_no-extra-fixed-base-precomputation\""
        <<",\"cpu_comparison_measured\":false,\"protocol_performance_measured\":false,\"diagnostic_only\":false,\"exact_match\":true}\n";
    std::ofstream file(output/"run-result.json");file<<report.str();file.close();require(bool(file),"failed to write benchmark report");
    std::cout<<report.str()<<"MSM_PREPARED_TARGET_MEDIAN seconds="<<steady_s<<" samples="<<options.samples
        <<" warmups="<<options.warmups<<"\nMSM_CPU_NPU_MATCH count="<<n<<std::endl;
    return 0;
}
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
    bool have_mode=false,have_cases=false,have_reuse=false;
    for(int i=3;i<argc;++i){
        const std::string_view flag(argv[i]);
        require(i+1<argc,"missing option value");
        const char* value=argv[++i];
        if(flag=="--mode"){
            require(!have_mode,"duplicate option");have_mode=true;
            require(std::string_view(value)=="one-shot"||std::string_view(value)=="prepared","mode must be one-shot or prepared");
            options.prepared=std::string_view(value)=="prepared";
        }else if(flag=="--scalar-cases"){
            require(!have_cases,"duplicate option");have_cases=true;
            require(*value,"empty scalar-case manifest path");options.scalar_cases=value;
        }else if(flag=="--reuse-count"){
            require(!have_reuse,"duplicate option");have_reuse=true;options.reuse_count=parse_count(value,true);
        }else{
            require(flag=="--warmups"||flag=="--samples"||flag=="--window-bits"||flag=="--count","unknown option");
            const bool samples=flag=="--samples",window=flag=="--window-bits",input_count=flag=="--count";
            require(input_count?!have_count:(window?!have_window:(samples?!have_samples:!have_warmups)),"duplicate option");
            const uint32_t count=parse_count(value,samples||input_count);
            if(window){require(count==0||(count>=2&&count<=20),"window bits must be 0 or 2..20");options.window_bits=count;have_window=true;}
            else if(input_count){options.count=count;have_count=true;}
            else if(samples){options.samples=count;have_samples=true;}
            else{options.warmups=count;have_warmups=true;}
        }
    }
    require(options.prepared||(!have_cases&&!have_reuse),"scalar-cases and reuse-count require prepared mode");
    if(options.prepared&&!have_samples)options.samples=3;
    const std::filesystem::path root(argv[1]),output(argv[2]);
    std::cout<<"MSM_SETUP_BEGIN mode="<<(options.prepared?"prepared":"one-shot")
        <<" warmups="<<options.warmups<<" samples="<<options.samples<<std::endl;
    std::filesystem::copy_file("/proc/self/exe",output/"loaded-executable.elf",std::filesystem::copy_options::overwrite_existing);
    std::ifstream maps("/proc/self/maps");std::ofstream(output/"process-maps.txt")<<maps.rdbuf();
    std::ofstream(output/"process-identity.txt")<<"pid="<<getpid()<<"\nexe="<<std::filesystem::read_symlink("/proc/self/exe")<<"\n";
    if(options.prepared)return run_prepared_benchmark(options,root,output);
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
        <<"{\"execution_mode\":\"one-shot\",\"count\":"<<n<<",\"samples\":"<<options.samples<<",\"warmup\":"<<options.warmups<<",\"window_bits\":"<<p.window_bits<<",\"windows\":"<<p.windows
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
