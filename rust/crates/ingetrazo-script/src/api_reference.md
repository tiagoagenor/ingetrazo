# Reference

Numbers are Rhai integers or floats; `Vec3` is a point or a vector (its
components are `f32`, as the model's). Anywhere a point is expected a
three-number array `[x, y, z]` works too. "Entities" means one handle or
an array of handles (`Face`, `Edge`, `Group`).

## Points
`vec3(x, y, z)`, `v.x v.y v.z` (readable and writable), `a + b`, `a - b`,
`-a`, `a * k`, `k * a`, `a / k`, `a == b`, `dot(a, b)`, `cross(a, b)`,
`v.length()`, `v.normalized()`, `distance(a, b)`.

## Reading the model (the active mesh)
* `faces()`, `edges()` -> arrays of handles; `vertices()` -> array of `Vec3`.
* `face_count()`, `edge_count()`, `vertex_count()`.
* Face: `f.points()`, `f.normal()`, `f.area()`, `f.layer`, `f.color`
  (array or `()`), `f.opacity`, `f.material` (name or `()`), `f.hidden`,
  `f.attr("key")`, `f.alive`.
* Edge: `e.a`, `e.b`, `e.length()`, `e.soft`, `e.hidden`, `e.layer`, `e.alive`.
* `groups()` -> the top-level groups and components; Group: `g.name`,
  `g.path`, `g.layer`, `g.hidden`, `g.locked`, `g.is_component`,
  `g.children()`, `g.faces()`, `g.edges()`, `g.alive`; `find_group("name")`
  (depth-first, `()` if none).
* `layers()`, `materials()`, `material("name")` (a map or `()`),
  `dimensions()`, `text_labels()`, `guides()`, `section_planes()` -> arrays
  of names / maps.
* `selection()` -> the handles the host set as the UI selection.

## Changing the model (each call one command; the run is one undo step)
* Drawing, through the edits planner (it welds, splits and detects faces
  as the tools do): `add_edge(a, b)`, `add_polyline(points)`,
  `add_polygon(points)` (closed; the face inside is detected).
* Explicit faces: `add_face(points)`, `add_face(points, attrs_map)` where
  `attrs_map` is stamped on the face (`#{color: [r, g, b], layer: "Muros"}`,
  colours are floats 0-1, the Paint idiom). Returns the face handle.
* `erase(entities)`, `flip_faces(faces)`.
* `translate(entities, delta)`, `rotate(entities, center, axis, degrees)`,
  `scale(entities, center, factor)` (a number, or a `Vec3` per axis).
  Faces and edges move their vertices; groups move as a whole.
* `make_group(entities)`, `make_group(entities, name)`,
  `make_component(entities, name)` -> the new group handle (an entity list
  of groups nests them); `rename_group(g, name)`, `delete_group(g)`,
  `explode_group(g)`, `lock_group(g, bool)`.
* `hide(entities)`, `unhide(entities)`, `assign_layer(entities, "name")`,
  `set_color(faces, r, g, b)`, `clear_color(faces)`, `set_opacity(faces, a)`,
  `set_material(faces, "name")` (a material that exists in the document).
* Annotations: `add_dimension(a, b, offset)`, `add_text_label(anchor, text)`.

`print(x)` / `debug(x)` go to the run's output. A failing call (a stale
handle, a command that declines) stops the script with its message and
the run is rolled back.
