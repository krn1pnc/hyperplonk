#include "square_probe.h"
#include "runtime_audit.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace hyperplonk::ascend910b3::test;
void check(aclError e, const char* what) {
    if (e != ACL_SUCCESS) throw std::runtime_error(std::string(what) + ": " + std::to_string(e));
}
void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}
struct Corpus {
    uint32_t n = 0, count = 0;
    std::vector<uint32_t> data;
    explicit Corpus(const std::filesystem::path& path) {
        std::ifstream f(path, std::ios::binary);
        uint32_t magic = 0;
        f.read(reinterpret_cast<char*>(&magic),4);
        f.read(reinterpret_cast<char*>(&n),4);
        f.read(reinterpret_cast<char*>(&count),4);
        require(magic == 0x53515032 && (n == 22 || n == 32) && count >= 257 && count <= 1024,
                "invalid bounded fixture header");
        data.resize(size_t(n)*count*3);
        f.read(reinterpret_cast<char*>(data.data()),data.size()*4);
        require(bool(f) && f.peek() == std::char_traits<char>::eof(), "invalid fixture extent");
    }
    uint32_t at(uint32_t sample, uint32_t policy, uint32_t digit) const {
        return data[(size_t(sample % count)*3+policy)*n+digit];
    }
};
bool retain_runtime_resources = false;
aclError drain(aclrtStream stream) noexcept {
    const aclError status = aclrtSynchronizeStream(stream);
    if (status != ACL_SUCCESS) retain_runtime_resources = true;
    return status;
}
struct Device {
    void* ptr = nullptr;
    explicit Device(size_t bytes) { check(aclrtMalloc(&ptr,bytes,ACL_MEM_MALLOC_NORMAL_ONLY),"malloc"); }
    ~Device() { if (ptr && !retain_runtime_resources) aclrtFree(ptr); }
};
struct Runtime {
    aclrtStream stream = nullptr;
    bool initialized = false, device = false;
    Runtime() {
        check(aclInit(nullptr),"aclInit"); initialized = true;
        try {
            check(aclrtSetDevice(0),"set device"); device = true;
            check(aclrtCreateStream(&stream),"create stream");
        } catch (...) { cleanup(); throw; }
    }
    void cleanup() noexcept {
        if (retain_runtime_resources) return;
        if (stream) {
            if (drain(stream) != ACL_SUCCESS) {
                std::cerr<<"SQUARE_PROBE_DRAIN_FAILED_RESOURCES_RETAINED\n";
                return;
            }
            aclrtDestroyStream(stream); stream=nullptr;
        }
        if (device) { aclrtResetDevice(0); device=false; }
        if (initialized) { aclFinalize(); initialized=false; }
    }
    ~Runtime() { cleanup(); }
};
uint32_t word(const uint8_t* p, size_t offset, uint32_t width=4) {
    uint32_t v=0; std::memcpy(&v,p+offset,width); return v;
}
struct Layout {
    uint32_t l,n,row,banks,acc,tmp,arena,extra,field,element,size;
    Layout(uint32_t lanes,uint32_t digits,uint32_t mode):l(lanes),n(digits),row(lanes*4) {
        bool packed=l==64;
        banks=(packed?62:124)*row;
        acc=banks+(packed?5:8)*32*row;
        tmp=acc+65*row;
        arena=tmp+16*row;
        element=(mode==2 && packed)?2:4;
        field=n*l*element;
        extra=mode==2?(packed?100352:93696):0;
        size=arena+extra;
    }
    size_t result(uint32_t mode,bool alias) const {
        if(mode==2 && l==64) return arena+(alias?0:3)*field;
        return banks+(alias?0:(mode==2?3:2))*32*row;
    }
};
bool in(size_t x,size_t begin,size_t bytes) { return x>=begin && x<begin+bytes; }
void validate(const Layout& a,const Corpus& corpus,const std::vector<uint8_t>& debug,
              uint32_t count,uint32_t mode,bool canonical,bool alias,bool last_only) {
    uint32_t tiles=(count+a.l-1)/a.l;
    size_t result=a.result(mode,alias);
    for(uint32_t tile=0;tile<tiles;++tile) for(uint32_t pass=0;pass<2;++pass) {
        size_t rec=tile*2+pass;
        const uint8_t* before=debug.data()+64+rec*2*a.size;
        const uint8_t* after=before+a.size;
        for(uint32_t lane=0;lane<a.l;++lane) {
            uint32_t index=tile*a.l+lane;
            bool active=index<count && (!last_only || index==count-1);
            uint32_t sample=active?(index+pass*count)%corpus.count:0;
            for(uint32_t d=0;d<a.n;++d) {
                uint32_t expected=corpus.at(sample,canonical?1:2,d);
                uint32_t actual=word(after,result+(size_t(d)*a.l+lane)*a.element,a.element);
                require(actual==expected,"exact square mismatch mode="+std::to_string(mode)+
                    " L="+std::to_string(a.l)+" count="+std::to_string(count)+
                    " pass="+std::to_string(pass)+" lane="+std::to_string(lane)+
                    " digit="+std::to_string(d));
            }
            require(word(after,a.acc+size_t(2*a.n)*a.row+lane*4)==0,"high guard not zero");
        }
        for(size_t x=0;x<a.size;++x) {
            bool mutable_byte=in(x,a.acc,(2*a.n+1)*a.row) || in(x,a.tmp,6*a.row) ||
                in(x,a.banks+(a.l==64?3:6)*32*a.row,32*a.row) ||
                in(x,result,a.field);
            if(mode==2 && a.l==64)
                mutable_byte=mutable_byte || in(x,a.banks,32*a.row) ||
                    in(x,a.banks+2*32*a.row,32*a.row);
            if(!mutable_byte) require(before[x]==after[x],
                "preserved byte modified offset="+std::to_string(x)+" mode="+std::to_string(mode));
        }
        // All unused Fr accumulator rows are poisoned and remain untouched.
        for(uint32_t d=2*a.n+1;d<65;++d) for(uint32_t lane=0;lane<a.l;++lane)
            require(word(after,a.acc+d*a.row+lane*4)==0x24681,"Fr unused row poison changed");
    }
    require(std::all_of(debug.begin(),debug.begin()+64,[](uint8_t x){return x==0xa5;}),"debug prefix guard");
    require(std::all_of(debug.end()-64,debug.end(),[](uint8_t x){return x==0xa5;}),"debug suffix guard");
}
void no_resources(const RuntimeCounts& c) {
    require(c.malloc_calls==0 && c.free_calls==0 && c.host_malloc_calls==0 &&
        c.host_free_calls==0 && c.h2d_calls==0 && c.other_copy_calls==0 &&
        c.unexpected_copy_calls==0 && c.device_sync_calls==0,"unexpected runtime resource");
}
void run(Runtime& runtime,const Corpus& corpus,uint32_t count,uint32_t lanes,uint32_t mode,
         bool canonical,bool alias,bool last_only) {
    Layout a(lanes,corpus.n,mode);
    size_t records=size_t((count+lanes-1)/lanes)*2;
    std::vector<uint8_t> input(std::max<size_t>(64,records*a.field)+384,0xa5);
    for(size_t rec=0;rec<records;++rec) for(uint32_t d=0;d<a.n;++d)
        for(uint32_t lane=0;lane<lanes;++lane) {
            uint32_t index=uint32_t(rec/2)*lanes+lane;
            bool active=index<count && (!last_only || index==count-1);
            uint32_t sample=active?(index+uint32_t(rec%2)*count)%corpus.count:0;
            uint32_t value=corpus.at(sample,0,d);
            std::memcpy(input.data()+64+rec*a.field+(size_t(d)*lanes+lane)*a.element,&value,a.element);
        }
    for(uint32_t lane=0;lane<lanes;++lane) {
        uint32_t sentinel=0x57913000+lane;
        std::memcpy(input.data()+64+records*a.field+lane*4,&sentinel,4);
    }
    std::vector<uint8_t> debug(std::max<size_t>(64,records*2*a.size)+128,0xa5);
    std::vector<uint8_t> input_back(input.size());
    Device di(input.size()),dd(debug.size());
    check(aclrtMemcpy(di.ptr,input.size(),input.data(),input.size(),ACL_MEMCPY_HOST_TO_DEVICE),"input upload");
    check(aclrtMemcpy(dd.ptr,debug.size(),debug.data(),debug.size(),ACL_MEMCPY_HOST_TO_DEVICE),"debug upload");
    RuntimeCounts enqueue;
    begin_runtime_audit(nullptr,0);
    try {
        launch_square_probe(runtime.stream,static_cast<uint8_t*>(di.ptr)+64,
            static_cast<uint8_t*>(dd.ptr)+64,count,lanes,mode,canonical,alias);
        enqueue=end_runtime_audit();
    } catch(...) { end_runtime_audit(); drain(runtime.stream); throw; }
    // Always drain actual submitted work before any scratch can be freed, even if auditing fails.
    RuntimeCopyAllowance allowances[]={{dd.ptr,debug.data(),debug.size(),false},
                                     {di.ptr,input_back.data(),input_back.size(),false}};
    begin_runtime_audit(allowances,2);
    aclError drained=drain(runtime.stream);
    if (drained != ACL_SUCCESS) {
        end_runtime_audit();
        check(drained,"actual kernel drain; borrowed resources retained");
    }
    aclError first=aclrtMemcpy(debug.data(),debug.size(),dd.ptr,debug.size(),ACL_MEMCPY_DEVICE_TO_HOST);
    aclError second=aclrtMemcpy(input_back.data(),input_back.size(),di.ptr,input_back.size(),ACL_MEMCPY_DEVICE_TO_HOST);
    auto final=end_runtime_audit();
    check(drained,"actual kernel drain");
    no_resources(enqueue);
    require(enqueue.d2h_calls==0 && enqueue.stream_sync_calls==0,"enqueue copy/sync");
    check(first,"debug readback"); check(second,"input readback"); no_resources(final);
    require(final.d2h_calls==2 && final.d2h_bytes==debug.size()+input_back.size() &&
            final.stream_sync_calls==1,"final resource counts");
    require(input_back==input,"external input/guards changed");
    validate(a,corpus,debug,count,mode,canonical,alias,last_only);
}
}
int main(int argc,char** argv) {
    if(argc==2 && std::string(argv[1])=="--help") {
        std::cout<<"Usage: square_probe FIXTURES OUTPUT\n"
            "FIXTURES contains fr.bin/fq.bin generated by square_probe_reference.py OUTPUT.\n"
            "Runs real private radix-12 Fr/Fq square and actual G1 adapters on device 0;\n"
            "OUTPUT is a JSON evidence file. No fixture generation or CPU execution fallback.\n";
        return 0;
    }
    if(argc!=3) { std::cerr<<"Usage: square_probe FIXTURES OUTPUT\n"; return 2; }
    try {
        Corpus fr(std::filesystem::path(argv[1])/"fr.bin"),fq(std::filesystem::path(argv[1])/"fq.bin");
        Runtime runtime;
        uint32_t executions=0;
        const uint32_t counts[]={0,1,47,48,49,63,64,65,97,129,257};
        for(uint32_t mode=0;mode<3;++mode) for(uint32_t lanes:{48u,64u})
            for(bool canonical:{false,true}) {
                if(mode==2 && canonical) continue; // Production G1 policy is redundant only.
                for(bool alias:{false,true}) for(bool last:{false,true}) for(uint32_t count:counts) {
                    run(runtime,mode==0?fr:fq,count,lanes,mode,canonical,alias,last);
                    ++executions;
                }
            }
        std::ofstream out(argv[2]);
        out<<"{\"status\":\"PASS\",\"actual_launches\":"<<executions
           <<",\"enqueue_alloc_copy_sync\":0,\"readbacks_per_launch\":2,"
             "\"drains_per_launch\":1,\"exact_private_policies\":true,\"device\":0}\n";
        require(bool(out),"evidence write failed");
        std::cout<<"PASS actual square probe launches="<<executions<<"\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"FAIL square probe: "<<e.what()<<"\n"; return 1;
    }
}
