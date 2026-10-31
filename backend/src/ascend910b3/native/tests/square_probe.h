#pragma once
#include <acl/acl.h>
#include <cstdint>
namespace hyperplonk::ascend910b3::test {
// Private test launcher: mode 0/1 = Fr/Fq primitive, mode 2 = actual G1 adapter.
// Each tile has two changing input records; debug contains before/after arenas.
void launch_square_probe(aclrtStream stream, uint8_t* input, uint8_t* debug,
    uint32_t count, uint32_t lanes, uint32_t mode, bool canonical, bool alias);
}
