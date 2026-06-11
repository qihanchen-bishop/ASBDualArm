"""Script to check USD file structure."""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics, UsdGeom

usd_path = '/workspace/isaaclab/source/OpenH_Data/envs/onlyorgan.usd'
stage = Usd.Stage.Open(usd_path)

print('=== USD Structure ===')
print(f'File: {usd_path}\n')

rigid_bodies = []
deformables = []
meshes = []

for prim in stage.Traverse():
    prim_path = str(prim.GetPath())
    prim_type = prim.GetTypeName()
    
    has_rigid = prim.HasAPI(UsdPhysics.RigidBodyAPI)
    # Check for deformable
    applied_schemas = prim.GetAppliedSchemas()
    has_deformable = any('Deformable' in str(api) for api in applied_schemas)
    
    if has_rigid:
        rigid_bodies.append((prim_path, prim_type))
    if has_deformable or 'Deformable' in prim_type:
        deformables.append((prim_path, prim_type))
    if prim_type == 'Mesh':
        meshes.append(prim_path)

print('=== Rigid Bodies ===')
for path, ptype in rigid_bodies:
    print(f'  {path} ({ptype})')

print('\n=== Potential Deformables ===')
for path, ptype in deformables:
    print(f'  {path} ({ptype})')

print(f'\n=== Total Meshes: {len(meshes)} ===')
for path in meshes[:10]:
    print(f'  {path}')
if len(meshes) > 10:
    print(f'  ... and {len(meshes) - 10} more')

simulation_app.close()
