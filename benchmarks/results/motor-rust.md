# Motor Rust experimental — mediciones (0.5.9, Bloque R)

Máquina de Marco (Linux, 16 hilos). `scripts/bench_engine.py ~/Descargas/plaza.igz --no-profile`,
una vez con `INGETRAZO_RUST=0` y otra con `=1`. Segundos.

## R1 — línea base (17-10, `c9f547f`)

| Operación | Python | Rust | Nota |
|---|---:|---:|---|
| abrir | 1,90 | 1,83 | objetos Python: no gana sin malla en Rust |
| guardar | 0,91 | 0,86 | |
| exportar STL | 3,35 | 3,21 | |
| exportar OBJ | 4,80 | 4,58 | |
| collect_geometry (láminas) | 3,96 | 3,75 | |
| líneas ocultas, planta | 150,4 | **5,0** | HLR (13-10) |
| líneas ocultas, iso | 140,9 | **6,0** | |
| triangular todo | 1,50 | 1,49 | |
| orientar el grupo más grande | 6,80 | 6,81 | |
| intersecar todo con todo (82 049 cruces) | 257 | 257 | 95 % en `_touching` (cajas, NumPy) |
| aplicar 2 000 cruces | 4,2 | 4,5 | `_cycle_adjacency`, `edits._pick` |

## R2

- **Intersecar — fase de cajas** (`touching_boxes`, rejilla x/y + la misma prueba en f64):
  todo con todo en plaza **266 → 8,7 s**, los 82 049 segmentos idénticos (mismo proceso, mismo orden).
  `tests/test_intersect_native.py` caza 5/5 sabotajes (eps, lista de cajas grandes, i=j, consulta sin eps en x / y).
  Ojo: entre ejecuciones Python da 82 033–82 049 cruces porque la selección es un `set` (su orden cambia): comparar siempre en el mismo proceso.
- **Rayos de paridad** (`ParityTris`: BVH propio + el mismo Möller–Trumbore en f64), base de orientar y de la reconstrucción de empujar/tirar:
  orientar las 892 mallas de plaza **31,3 → 8,0 s** (ya orientadas) y **34,4 → 8,5 s** con el 20 % de caras volteadas al azar (3 752 volteos, 809 tabiques): resultado idéntico cara por cara.
  `tests/test_parity_native.py` caza 7/7 sabotajes (t mínimo, t detrás, u sin eps, cajas sin margen, máscara, cara excluida). La máscara por cara ya no se copia en Python: se pasa la base y la cara a excluir (`skip`).

## Interacción (plaza, Python, `bench_interaction.py --repeat 1`, ms por evento)

| gesto | mediana | p95 |
|---|---:|---:|
| órbita / encuadre / zoom | ~16 | ~23 |
| pasar con Selección | 17,9 | 29,8 |
| pasar con Línea (inferencias) | 33,0 | 52,0 |
| clic seleccionar | 44,9 | 77,5 |
| mover un edificio | 42,4 | 222,4 |
| deshacer | 98,9 | — |

(El banco se quedaba colgado al cerrar en el diálogo «¿descartar cambios?»: arreglado.)

## Pendiente fuera del motor Rust (camino Python)

- **Aplicar los cruces de Intersecar con todo seleccionado** (#541): 2 000 cruces 4,2 s, 8 000 cruces 112 s (crece mucho más que lineal: `_cycle_adjacency` rehace la adyacencia de las aristas cercanas por cada segmento) y además **falla**: «degenerate edge: endpoints weld to one vertex» y se deshace el paso entero, aunque los segmentos ya pasan el filtro `_WELD_LENGTH` del menú (los extremos se sueldan después de partir otras aristas). Igual con los dos motores.
- `Face.normal()` se recalcula ~1 M veces en 4 empujes sobre el terreno de plaza: memorizarla por escaneo ganaría mucho en los dos motores (cambio del camino Python: no en la 0.5.9).

## Migración a Rust, fase 1 y fase 2 (rama `rust-migration`, 18-10-2026)

Máquina de Tiago (Windows 11, 28 hilos). Modelos sintéticos de `scripts/bench_models.py` (city-M: 73 102 caras; city-L: 331 124), mediana de 5 salvo donde se indica. Plan y criterios: `docs/rust-migration-plan.md`.

| Operación | Antes | Después | Nota |
|---|---:|---:|---|
| abrir hasta el primer dibujo, city-M (motor encendido, Windows) | 4,34 s | 0,72 s | mallas ligeras dibujadas desde arreglos: `qt_hypot` ya reconoce MSVC |
| abrir hasta el primer dibujo, city-L | 16,31 s | 1,64 s | |
| `lazy_mesh.plan`, rejilla 300×300 (90 000 caras) | 407,5 ms | 155,6 ms | `plan_topology` + ids de curva/capa como arreglos |
| aplicar 2 000 cruces de Intersecar, city-M | 10,2 s | 0,54 s | grafo de ciclos vivo (4,3 s) y filtro de cajas incremental |
| aplicar 8 000 cruces de Intersecar, city-M | 195,5 s | 2,95 s | #541: los dos costes cuadráticos del planificador (Python) |
| `_build_graph`, 300 paredes cortas (planta) | 70,2 ms | 3,8 ms | `arrangement_splits` |
| `_build_graph`, 300 segmentos al azar que se cruzan | 178 ms | 66 ms | lo que queda es el grafo en Python |
| `section_loops`, 35 000 cuerdas | 683,7 ms | 103,6 ms | |
| Intersecar ▸ Con la selección (encontrar), city-M | 10,5 s | 2,3 s | `PolySet` |
| Intersecar ▸ Con la selección (encontrar), city-L | 44,6 s | 10,0 s | |
| `_conform_t_junctions`, muro con 2 000 puertas | 3 140 ms | 4 ms | |


El «degenerate edge» con 8 000 cruces no se reprodujo en city-M (la plaza no está en esta máquina): sigue pendiente de comprobar con `plaza.igz`.

## Migración a Rust, fase 3 (formatos y georreferencia)

Misma máquina, prioridad baja. Los archivos exportados son idénticos byte a byte a los del camino Python (COLLADA: salvo sus marcas `<created>`/`<modified>`, que cambian entre dos ejecuciones también sin motor).

| Operación | Antes | Después | Nota |
|---|---:|---:|---|
| fachada con 2 000 ventanas, `_maximal_holes` | 59,3 s | 0,11 s | A1, Python |
| simplificar malla, 6 541 vértices | 1,12 s | 0,68 s | A2, Python |
| LZW, 4,2 MB de imagen | 843 ms | 19 ms | L1 |
| fusionar 45 000 triángulos coplanares | 3,05 s | 0,40 s | K6 |
| exportar STL, city-M | 9,09 s | 0,94 s | E1 |
| exportar STL, city-L | 43,13 s | 4,37 s | E1 |
| exportar OBJ, city-M | 9,10 s | 4,75 s | E1 |
| exportar OBJ, city-L | 44,48 s | 22,35 s | E1 |
| exportar COLLADA, city-M | 10,35 s | 9,41 s | E1 |
| importar STL de 200 000 triángulos | 14,3 s | 5,97 s / 0,12 s | sin motor (una pasada) / con motor (malla ligera) |
| importar GLB de 736 672 triángulos (`build_mesh`) | 17,88 s | 2,48 s | I1 |
| DXF, 20 000 3DFACE coplanares | 16,63 s | 2,69 s | A3 |

Rechazados con números: leer el texto de un OBJ es el 4 % de importarlo (0,68 de 16,5 s, city-M); la exportación .skp no tiene llamador; la rejilla de terreno tarda 0,46 s en su tope de 180 × 180.
