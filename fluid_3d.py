import taichi as ti

lin_iters = 20

n = 128
source_size = 20
v = ti.Vector.field(3, float, shape=(n, n, n))
new_v = ti.Vector.field(3, float, shape=(n, n, n))
new_v_aux = ti.Vector.field(3, float, shape=(n, n, n))
dens = ti.field(float, shape=(n, n, n))
new_dens = ti.field(float, shape=(n, n, n))
new_dens_aux = ti.field(float, shape=(n, n, n))
dx = 1.0 / n
dt = 0.1

stagger = ti.Vector([0.5, 0.5, 0.5])

@ti.func
def sample(p):
    p = clamp(p)
    p_grid = p * n - stagger
    I = ti.cast(ti.floor(p_grid), ti.i32)
    return I

@ti.func
def clamp(p):
    for d in ti.static(range(p.n)):
        p[d] = min(1 - 1e-4 - dx + stagger[d] * dx, max(p[d], stagger[d] * dx))
    return p

@ti.func
def sample_trilinear(x, p):
    p = clamp(p)

    p_grid = p * n - stagger

    I = ti.cast(ti.floor(p_grid), ti.i32)
    f = p_grid - I
    g = 1.0 - f
    res = x[I] * 0.0
    for i, j, k in ti.ndrange(2, 2, 2):
        res += (i*f[0]+(1-i)*g[0])*(j*f[1]+(1-j)*g[1])*(k*f[2]+(1-k)*g[2])*x[I + ti.Vector([i, j, k])]

    return res


@ti.func
def backtrace(I, dt):
    p = (I + stagger) * dx
    v1 = v[sample(p)]
    p1 = p - 0.5 * dt * v1
    v2 = v[sample(p1)]
    p2 = p - 0.75 * dt * v2
    v3 = v[sample(p2)]
    p -= dt * (2 / 9 * v1 + 1 / 3 * v2 + 4 / 9 * v3)
    return p

@ti.func
def semi_lagrangian(x, new_x, dt):
    for I in ti.grouped(x):
        new_x[I] = sample_trilinear(x, backtrace(I, dt))

@ti.kernel
def maccormack(x : ti.template(), new_x : ti.template(), new_x_aux : ti.template(), dt : float):
    semi_lagrangian(x, new_x, dt)
    semi_lagrangian(new_x, new_x_aux, -dt)

    for I in ti.grouped(x):
        new_x[I] = new_x[I] + 0.5 * (x[I] - new_x_aux[I])

@ti.func
def sample_min_dens(x, p):
    I = sample(p)
    res = x[I]
    for i, j, k in ti.ndrange(2, 2, 2):
        if (res > x[I + ti.Vector([i, j, k])]):
            res = x[I + ti.Vector([i, j, k])]
    return res

@ti.func
def sample_max_dens(x, p):
   I = sample(p)
   res = x[I]
   for i, j, k in ti.ndrange(2, 2, 2):
        if (res < x[I + ti.Vector([i, j, k])]):
            res = x[I + ti.Vector([i, j, k])]
   return res

@ti.func
def sample_min_v(x, p):
    I = sample(p)
    res = x[I]
    for d in ti.static(range(3)):
        for i, j, k in ti.ndrange(2, 2, 2):
            if (res[d] > x[I + ti.Vector([i, j, k])][d]):
                res[d] = x[I + ti.Vector([i, j, k])][d]
    return res

@ti.func
def sample_max_v(x, p):
    I = sample(p)
    res = x[I]
    for d in ti.static(range(3)):
        for i, j, k in ti.ndrange(2, 2, 2):
            if (res[d] < x[I + ti.Vector([i, j, k])][d]):
                res[d] = x[I + ti.Vector([i, j, k])][d]
    return res

@ti.kernel
def mc_clipping_v():
    for I in ti.grouped(v):
        source_pos = backtrace(I, dt)
        min_val = sample_min_v(v, source_pos)
        max_val = sample_max_v(v, source_pos)
            
        if any(new_v[I] < min_val) or any(new_v[I] > max_val):
            new_v[I] = sample_trilinear(v, source_pos)

@ti.kernel
def mc_clipping_dens():
    for I in ti.grouped(dens):
        source_pos = backtrace(I, dt)
        min_val = sample_min_dens(dens, source_pos)
        max_val = sample_max_dens(dens, source_pos)
            
        if new_dens[I] < min_val or new_dens[I] > max_val:
            new_dens[I] = sample_trilinear(dens, source_pos)

@ti.kernel
def update(x : ti.template(), new_x: ti.template()):
    for I in ti.grouped(x):
        x[I] = new_x[I]

div = ti.field(float, shape=(n, n, n))
p = ti.field(float, shape=(n, n, n))

@ti.func
def lin_solve(x : ti.template(), x0 : ti.template(), a : float, c : float):
    for k in ti.static(range(lin_iters)):
        for I in ti.grouped(x):
            x[I] = (x0[I] + a * (x[sample((I + ti.Vector([1.0, 0.0, 0.0])) * dx)] + x[sample((I + ti.Vector([-1.0, 0.0, 0.0]) * dx))]
                         + x[sample((I + ti.Vector([0.0, 1.0, 0.0])) * dx)] + x[sample((I + ti.Vector([0.0, -1.0, 0.0]) * dx))]
                         + x[sample((I + ti.Vector([0.0, 0.0, 1.0])) * dx)] + x[sample((I + ti.Vector([0.0, 0.0, -1.0]) * dx))])) / c

@ti.kernel
def project(a : ti.template()):
    for I in ti.grouped(a):
        div[I] = -(a[sample((I + ti.Vector([1.0, 0.0, 0.0])) * dx)][0] - a[sample((I + ti.Vector([-1.0, 0.0, 0.0]) * dx))][0]
                     + a[sample((I + ti.Vector([0.0, 1.0, 0.0])) * dx)][1] - a[sample((I + ti.Vector([0.0, -1.0, 0.0]) * dx))][1]
                     + a[sample((I + ti.Vector([0.0, 0.0, 1.0])) * dx)][2] - a[sample((I + ti.Vector([0.0, 0.0, -1.0]) * dx))][2]) / (2.0 * n)
        p[I] = 0.0
    
    lin_solve(p, div, 1.0, 6.0)

    for I in ti.grouped(a):
        v[I][0] -= n * (p[sample((I + ti.Vector([1.0, 0.0, 0.0])) * dx)] - p[sample((I + ti.Vector([-1.0, 0.0, 0.0]) * dx))]) / 2.0
        v[I][1] -= n * (p[sample((I + ti.Vector([0.0, 1.0, 0.0])) * dx)] - p[sample((I + ti.Vector([0.0, -1.0, 0.0]) * dx))]) / 2.0
        v[I][2] -= n * (p[sample((I + ti.Vector([0.0, 0.0, 1.0])) * dx)] - p[sample((I + ti.Vector([0.0, 0.0, -1.0]) * dx))]) / 2.0

@ti.kernel
def add_source():
    for i, j, k in ti.ndrange(source_size, source_size, source_size):
        dens[i + (n - source_size) // 2, j + source_size, k + (n - source_size) // 2] += 100.0 * dt
    for i, j, k in ti.ndrange(n, n, n):
        if ti.random() > 0.9:
            v[i, j, k] += ti.Vector([2.0 * ti.random() - 1.0, 2.0 * ti.random() - 0.5, 2.0 * ti.random() - 1.0])

@ti.kernel
def dens_reset():
    for I in ti.grouped(v):
        dens[I] = 0.0

def step():
    add_source()
    maccormack(v, new_v, new_v_aux, dt)
    mc_clipping_v()
    update(v, new_v)
    project(v)
    maccormack(dens, new_dens, new_dens_aux, dt)
    mc_clipping_dens()
    update(dens, new_dens)