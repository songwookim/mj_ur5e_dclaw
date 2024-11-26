# reference 
# 1: https://www.danaukes.com/work-blog/2024-03-11-double-pendulum-inverse/
# 2: https://mujoco.readthedocs.io/en/stable/computation/index.html#inverse-dynamics-and-optimization
# 3: https://alefram.github.io/posts/Basic-inverse-kinematics-in-Mujoco

import mujoco as mj
import mujoco.viewer
from robot_descriptions import ur5e_mj_description
import numpy as np
# import mediapy as media
model = mj.MjModel.from_xml_path(ur5e_mj_description.MJCF_PATH)
# other option : model = load_robot_description("ur5e_mj_description")  [from robot_descriptions.loaders.mujoco import load_robot_description]

data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera() 
mujoco.mjv_defaultFreeCamera(model, cam)
cam.distance = 1
opt = mj.MjvOption()                        # visualization options
spec = mj.MjSpec()    


#Put a position of the joints to get a test point
pi = np.pi
data.qpos = [3*pi/2, -pi/2, pi/2, 3*pi/2, 3*pi/2, 0]

#Inititial joint position
qpos0 = data.qpos.copy()

#Step the simulation.
mujoco.mj_forward(model, data)

#Use the last piece as an "end effector" to get a test point in cartesian coordinates
target = data.body('wrist_3_link').xpos.copy()

#Plot results
print("Results")
mujoco.mj_resetDataKeyframe(model, data, 1)
mujoco.mj_forward(model, data)
renderer = mujoco.Renderer(model)
init_point = data.body('wrist_3_link').xpos.copy()
renderer.update_scene(data, cam)
target_plot = renderer.render()

data.qpos = qpos0
mujoco.mj_forward(model, data)
result_point = data.body('wrist_3_link').xpos.copy()
renderer.update_scene(data, cam)
result_plot = renderer.render()

print("initial point =>", init_point)
print("Desire point =>", result_point, "\n")

images = {
    'Initial position': target_plot,
    ' Desire end effector position': result_plot,
}

# media.show_images(images)

#get the name of the joints and its limits
for j in range(len(data.qpos)):
    print("name part =>", data.jnt(j).name, "\n", 
          "limit =>", model.jnt_range[j], "\n")
    
# with mujoco.viewer.launch_passive(model, data) as viewer:
#     time_prev = data.time

#     while viewer.is_running():
#         data.qacc = data.qacc+100
#         mj.mj_step(model, data)
#     # qM : Moment of inertia (vecotr : 21x1)
#     # mj.mj_fullM()

#     # Pick up changes to the physics state, apply perturbations, update options from GUI.
    # viewer.sync()


    #Video Setup
DURATION = 4 #(seconds)
FRAMERATE = 60 #(Hz)
frames = []

#Reset state and time.
mujoco.mj_resetData(model, data)

#Init position.
# pi = np.pi
# data.qpos = [3*pi/2, -pi/2, pi/2, 3*pi/2, 3*pi/2, 0] #ENABLE if you want test circle

#Init parameters
jacp = np.zeros((3, model.nv)) #translation jacobian
jacr = np.zeros((3, model.nv)) #rotational jacobian
step_size = 0.5
tol = 0.01
alpha = 0.5
damping = 0.

#Get error.
end_effector_id = model.body('wrist_3_link').id #"End-effector we wish to control.
current_pose = data.body(end_effector_id).xpos #Current pose

goal = [0.49, 0.13, 0.59] #Desire position

x_error = np.subtract(goal, current_pose) #Init Error

def check_joint_limits(q):
    """Check if the joints is under or above its limits"""
    for i in range(len(q)):
        q[i] = max(model.jnt_range[i][0], min(q[i], model.jnt_range[i][1]))

def circle(t: float, r: float, h: float, k: float, f: float) -> np.ndarray:
    """Return the (x, y) coordinates of a circle with radius r centered at (h, k)
    as a function of time t and frequency f."""
    x = r * np.cos(2 * np.pi * f * t) + h
    y = r * np.sin(2 * np.pi * f * t) + k
    z = 0.5
    return np.array([x, y, z])

#Simulate
with mujoco.viewer.launch_passive(model, data) as viewer:
    time_prev = data.time
    while viewer.is_running():
        # while data.time < DURATION:
            
        goal = circle(data.time, 0.1, 0.5, 0.0, 0.5) #ENABLE to test circle.
        
        if (np.linalg.norm(x_error) >= tol):
            #Calculate jacobian
            mujoco.mj_jac(model, data, jacp, jacr, goal, end_effector_id)
            #Calculate delta of joint q
            n = jacp.shape[1]
            I = np.identity(n)
            product = jacp.T @ jacp + damping * I

            if np.isclose(np.linalg.det(product), 0):
                j_inv = np.linalg.pinv(product) @ jacp.T
            else:
                j_inv = np.linalg.inv(product) @ jacp.T

            delta_q = j_inv @ x_error 

            #Compute next step
            q = data.qpos.copy()
            q += step_size * delta_q
            
            # c(q,v) : coriolis, centrifugal, gravitational 
            print(f'c(q,v) : {data.qfrc_bias}')
            print(f'M(q) : {data.qM}')

            #Check limits
            check_joint_limits(data.qpos)
            
            #Set control signal
            data.ctrl = q 
            #Step the simulation.
            mujoco.mj_step(model, data)

            x_error = np.subtract(goal, data.body(end_effector_id).xpos)
            viewer.sync()
        # #Render and save frames.
        # if len(frames) < data.time * FRAMERATE:
        #     renderer.update_scene(data)
        #     pixels = renderer.render()
        #     frames.append(pixels)
        
#Display video.
# media.show_video(frames, fps=FRAMERATE)
# with open("data.html", "w") as file:
#     file.write(frames)