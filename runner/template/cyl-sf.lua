#!/usr/bin/env dgd-lua
--
-- cyl-sf.lua
-- Shock fitting boundary condition example simulation:
-- Cylinder in ideal air flow
-- Kyle Damm Jan 2015
-- PJ clean-up 2021
-- JB Oct 2022 convert to template for ML bow shock relation 
--
config.title = "Cylinder in ideal air flow with shock fitting boundary."
print(config.title)
config.dimensions = 2

-- problem parameters

dofile('input.lua')
-- loading
-- R1,R2,K1,K2,M

nsp, nmodes, gm = setGasModel('ideal-air-gas-model.lua')
p = 100.0e3
T = 300.0

Q = GasState:new{gm}
Q.p = p
Q.T = T
Q.massf = {air=1.0}
gm:updateThermoFromPT(Q)
gm:updateSoundSpeed(Q)
a_inf = Q.a
u_inf = M * a_inf

initial = FlowState:new{p=p/3.0, T=2.0*T/3.0, velx=0.0, vely=0.0}
inflow = FlowState:new{p=p, T=T, velx=u_inf, vely=0.0}

a  = Vector3:new{x=-R1, y=0.0}
b1 = Vector3:new{x=-R1, y=K1*R1}
b2 = Vector3:new{x=-K2*R2, y=R2}
c  = Vector3:new{x=0.0, y=R2}

d = Vector3:new{x=-2*R1, y=0}
e = Vector3:new{x=-2*R1, y=R1}
f = Vector3:new{x=-R1, y=2.5*R1}
g = Vector3:new{x=0.0, y=3.0*R1}

psurf = makePatch{north=Line:new{p0=g, p1=c},
		  east=Bezier:new{points={a,b1,b2,c}},
		  south=Line:new{p0=d, p1=a},
		  west=Bezier:new{points={d, e, f, g}}}

niv = 41
njv = 41

grid = StructuredGrid:new{psurface=psurf, niv=niv, njv=njv}

-- We can leave east and south as slip-walls
blk = FBArray:new{grid=grid, initialState=initial,
                  bcList={west=InFlowBC_ShockFitting:new{flowState=inflow},
                          north=OutFlowBC_Simple:new{}},
                  nib=8, njb=1}
identifyBlockConnections()
--
-- Set a few more config options
config.flux_calculator = "ausmdv"
config.gasdynamic_update_scheme = "backward_euler"
-- config.max_time = (R1*2)/u_inf * 20
config.max_time = (R1*2)/u_inf * 200
config.max_step = 4000000
config.cfl_value = 0.5
config.dt_init = 1e-7
config.dt_plot = config.max_time
config.grid_motion = "shock_fitting"
config.shock_fitting_delay = (R1*2)/u_inf  -- allow for one flow length
config.max_invalid_cells = 10
config.adjust_invalid_cell_data = true
config.report_invalid_cells = false
