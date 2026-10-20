// SPDX-License-Identifier: Apache-2.0
// One function Manifold's C API lacks: the per-triangle face ids of a
// MeshGL64 (Python's `Mesh64(..., face_id=...)`). `ManifoldMeshGL64` is
// `manifold::MeshGL64` seen through an opaque pointer (bindings/c/conv.h).
#include <cstddef>
#include <cstdint>

#include "manifold/manifold.h"

extern "C" void ingetrazo_meshgl64_set_face_id(void* mesh, const uint64_t* ids,
                                               size_t n) {
  auto* m = reinterpret_cast<manifold::MeshGL64*>(mesh);
  m->faceID.assign(ids, ids + n);
}
