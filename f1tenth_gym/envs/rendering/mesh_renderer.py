import sys
import numpy as np
from pathlib import Path
from pyqtgraph.opengl import MeshData, GLMeshItem

from .renderer import RenderSpec, EnvRenderer, ObjectRenderer

class MeshRenderer(ObjectRenderer):
    def __init__(self,
        env_renderer: EnvRenderer,
        render_spec: RenderSpec,
        map_origin: tuple[float, float],
        resolution: float,
        car_length: float,
        car_width: float,
        color: list[int] | None = None,
        wheel_size: float = 0.2
        ):
        super().__init__()
        import trimesh
        self.trimesh = trimesh
        self.car_length = car_length
        self.car_width = car_width
        self.wheel_size = wheel_size
        self.view = env_renderer.view
        
        self.load_obj()
        self.mesh_dict = self.get_mesh_dict()
        

        self.origin = map_origin
        self.resolution = resolution

        self.color = color
        self.pose = (0, 0, 0)
        self.steering = 0

        # Tire params need to be updated
        self.tire_width = 0.1
        self.tire_length = self.wheel_size
        
    
    def load_obj(self):
        # Load meshes (skip the ground plane)
        scene = self.trimesh.load(OBJ_PATH, process=False)
        self.parts = {n:m for n,m in scene.geometry.items() if n != "Plane.002"}

        # Compute wheel center
        centers = [self.parts[w].vertices.mean(axis=0) for w in WHEEL_NAMES]
        self.wheel_center = np.stack(centers).mean(axis=0)
        # print(self.wheel_center)
        
        all_vs = np.vstack([m.vertices for m in self.parts.values()])
        self.center_xy = np.array([all_vs[:,0].mean(), all_vs[:,1].mean(), 0], dtype=np.float32)
        
        self.car_pose = {
            "x": 0.0,      # position east
            "y": 0.0,      # position north
            "z": 0.0,      # height above ground plane
            "roll": 0.0,   # rotation about vehicle X axis
            "pitch": 0.0,  # rotation about vehicle Y axis
            "yaw": 0.0    # rotation about vehicle Z axis
        }
        
        # Build the 3D pose transform
        R = euler_to_matrix(self.car_pose["roll"] + 90, self.car_pose["pitch"], self.car_pose["yaw"])
        t = np.array([self.car_pose["x"], self.car_pose["y"], self.car_pose["z"]], dtype=np.float32)

        # Find the lowest Z after rotating about wheel center
        minz = np.inf
        for mesh in self.parts.values():
            V = mesh.vertices.astype(np.float32)
            Vc = V - self.wheel_center
            Vr = Vc @ R.T
            minz = min(minz, Vr[:,2].min())
        # Lift so wheels sit on Z=0, then add self.car_pose z
        self.z_lift = -minz + self.car_pose["z"]
        
    def update(self, obs: dict[str, np.ndarray], id: str):        
        state = obs[id]["std_state"].astype(float)
        self.car_pose['x'] = state[0]
        self.car_pose['y'] = state[1]
        self.car_pose['yaw'] = state[4] / np.pi * 180
        self.steering = state[2]   
        
    
    def get_mesh_dict(self):
        mesh_dict = {}
        for name, mesh in self.parts.items():
            V = mesh.vertices.astype(np.float32)
            F = mesh.faces.astype(np.int32)
            # center on wheels, rotate, then translate+lift
            Vc = V - self.wheel_center
            Vfinal = Vc + np.array([self.car_pose["x"], self.car_pose["y"], self.z_lift],dtype=np.float32)

            md = MeshData(vertexes=Vfinal, faces=F)
            item = GLMeshItem(meshdata=md, smooth=True,
                            shader='shaded', drawFaces=True, drawEdges=False)
            item.setColor(COLOR_MAP.get(name, DEFAULT_COLOR))
            item.setGLOptions('opaque')
            self.view.addItem(item)
            mesh_dict[name] = item
        return mesh_dict
    
    def render(self, scale=1.0):
        # Render each part with the full pose
        R = euler_to_matrix(self.car_pose["roll"] + 90, self.car_pose["pitch"], self.car_pose["yaw"])
        t = np.array([self.car_pose["x"], self.car_pose["y"], self.car_pose["z"]], dtype=np.float32)
        tx, ty = self.car_pose["x"], self.car_pose["y"]
        for name, mesh in self.parts.items():
            item = self.mesh_dict[name]
            item.resetTransform()
            # 2) center on wheels, apply rotation
            #    and apply global translation & lift in one go
            #    note translate is post-multiplied, so do them in reverse order:
            item.translate(-self.wheel_center[0],
                           -self.wheel_center[1],
                           -self.wheel_center[2])
            
            item.rotate(self.car_pose["roll"]+90, 1, 0, 0)
            item.rotate(self.car_pose["yaw"], 0, 0, 1)
            item.rotate(self.car_pose["pitch"], 0, 1, 0)
            item.translate(tx, ty, self.z_lift)



OBJ_PATH = Path(__file__).parent / "example_mesh_cyber.obj"
# Exact object‐name → RGBA color
COLOR_MAP = {
    "WBL_b1.004_Cylinder.035": (0.35, 0.35, 0.35, 1.0), # wheels
    "WBL_b1.003_Cylinder.034": (0.35, 0.35, 0.35, 1.0),
    "WBL_b1.002_Cylinder.033": (0.35, 0.35, 0.35, 1.0),
    "WBL_b1_Cylinder.020":     (0.35, 0.35, 0.35, 1.0),
    "WBL_b1.004_Cylinder.035_1": (0.35, 0.35, 0.35, 1.0),
    "bamper_l_Cube.001":       (1.0, 0.7, 0.2, 0.8), # turning light
    "bamper_d_Cube.005":       (0.7, 0.7, 0.7, 1.0), # wheel arch
    "body_Cube":               (1.0, 0.2, 0.2, 1.0), # tail light
    "body_Cube_1":             (1.0, 0.75, 0.65, 1.0), # front light
    "body_Cube_2":             (0.3, 0.3, 0.3, 0.3), # glass
    "bamper_l_Cube.001_1":     (0.85, 0.85, 0.87, 1.0), # body
    # add any others here...
}
DEFAULT_COLOR = (0.6, 0.6, 0.6, 1.0)

# Names of the four wheels (for centering)
WHEEL_NAMES = [
    "WBL_b1.004_Cylinder.035",
    "WBL_b1.003_Cylinder.034",
    "WBL_b1.002_Cylinder.033",
    "WBL_b1_Cylinder.020",
]

def euler_to_matrix(roll, pitch, yaw):
    """Build rotation matrix from roll, pitch, yaw (degrees)."""
    r, p, y = np.deg2rad([roll, pitch, yaw])
    Rx = np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]],dtype=np.float32)
    Ry = np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]],dtype=np.float32)
    Rz = np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]],dtype=np.float32)
    return Rz @ Ry @ Rx  # apply roll, then pitch, then yaw


