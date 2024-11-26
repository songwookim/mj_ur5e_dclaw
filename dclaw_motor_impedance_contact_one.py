import mujoco as mj
import mujoco.viewer
from robot_descriptions import ur5e_mj_description
from robot_descriptions.loaders.mujoco import load_robot_description
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

model = mj.MjModel.from_xml_path("./universal_robots_ur5e_with_dclaw/robel_sim/dclaw_motor/dclaw3xh.xml")

data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera() 
mujoco.mjv_defaultFreeCamera(model, cam)
cam.distance = 1
opt = mj.MjvOption()                        # visualization options
spec = mj.MjSpec()    


#Put a position of the joints to get a test point
pi = np.pi
# data.qpos = [3*pi/2, -pi/2, pi/2, 3*pi/2, 3*pi/2, 0]

#Inititial joint position
qpos0 = data.qpos.copy()

#Step the simulation.
mujoco.mj_forward(model, data)

#Use the last piece as an "end effector" to get a test point in cartesian coordinates
target = data.body('FFL12').xpos.copy()

#Plot results
print("Results")
mujoco.mj_resetDataKeyframe(model, data, 1)
mujoco.mj_forward(model, data)
renderer = mujoco.Renderer(model)
init_point = data.body('FFL12').xpos.copy()
renderer.update_scene(data, cam)
target_plot = renderer.render()

data.qpos = qpos0
mujoco.mj_forward(model, data)
result_point = data.body('FFL12').xpos.copy()
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



#Reset state and time.
mujoco.mj_resetData(model, data)

#Init parameters
jacp = np.zeros((3, model.nv)) #translation jacobian
jacr = np.zeros((3, model.nv)) #rotational jacobian

tol = 0.01
alpha = 0.5
damping = 0.25

#Get error.
end_effector_id = model.body('FFL12').id #"End-effector we wish to control.
current_pose = data.body(end_effector_id).xpos #Current pose

goal = [-0., -0.0, 0.1] #Desire position

x_error = np.subtract(goal, current_pose) #Init Error

def check_joint_limits(q):
    """Check if the joints is under or above its limits"""
    for i in range(len(q)):
        q[i] = max(model.jnt_range[i][0], min(q[i], model.jnt_range[i][1]))


#Simulate
i = 0
iter_ct = 0
x = data.xpos[3,:]
desired_stiffness= [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]
# desired_damping = [10, 10, 10], [10, 10, 10], [10, 10, 10]

desired_stiffness = 10
desired_damping = 3
desired_inertia = 30


force_list = []
f_imp_list = []
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.user_scn
    time_prev = data.time
    viewer.sync()
    with viewer.lock():
        viewer.user_scn.ngeom += 1

        # initial setting : geom, type, size, pos, rot, rgba
        mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom - 1], 
                mujoco.mjtGeom.mjGEOM_CAPSULE, 
                np.zeros(3), np.zeros(3), np.zeros(9), np.array([1, 0., 0., 1]))

        # change setting : change value of geom
        mujoco.mjv_connector(viewer.user_scn.geoms[viewer.user_scn.ngeom - 1], 
                mujoco.mjtGeom.mjGEOM_CAPSULE, 0.01,
                np.array(goal)-0.001,
                np.array(goal))
            
    FFtip_pos1 = data.site("FFtip").xpos
    FFtip_pos2 = np.zeros(3)
    FFtip_vel1 = np.zeros(3)
    FFtip_vel2 = np.zeros(3)
    FFtip_acc = np.zeros(3)
    time = 0
    while viewer.is_running():
        time += 1
    # if (np.linalg.norm(x_error) >= tol*15):
        #Calculate jacobian
        mujoco.mj_jac(model, data, jacp, jacr, data.site('FFtip').xpos, end_effector_id)
        # mj.mj_jacSite(model,data,jacp,jacr,0)
        #Calculate delta of joint q
        n = jacp.shape[1]
        I = np.identity(n)
        product = jacp.T @ jacp + damping * I

        # if np.isclose(np.linalg.det(product), 0):
        j_inv = np.linalg.pinv(product) @ jacp.T
        xvel = jacp@data.qvel 
        # else:
        #     j_inv = np.linalg.inv(product) @ jacp.T
        # j_inv = j_inv*20
        # x_error = x_error * 20
        # grad = 0.5* (j_inv @ x_error) # finger1: 5,6,7  finger 2: 9,10,11 finger 3:13,14,15

        mj.mj_rnePostConstraint(model,data)

        # -------------------------------------------
        dt = 0.0025
        FFtip_pos2 = data.site("FFtip").xpos.copy()
        FFtip_vel2 = (FFtip_pos2 - FFtip_pos1) / dt
        FFtip_acc = (FFtip_vel2 - FFtip_vel1) / dt

        FFtip_vel1 = FFtip_vel2
        FFtip_pos1 = FFtip_pos2
        # -------------------------------------------

        xacc = FFtip_acc        # data.body('FFL12').cacc[3:6]
        # print(FFtip_acc)
        F_imp = jacp.T @ (desired_inertia*xacc + desired_damping*xvel + desired_stiffness*x_error) 
        mj.mj_rnePostConstraint(model,data)
        data.qfrc_applied = F_imp
        
        #Check limits
        check_joint_limits(data.qpos)
        mj.mj_forward(model, data)

        with viewer.lock():
            # viewer.user_scn.ngeom += 1
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
            # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTSPLIT] = 1
            #  viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
            # mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom - 1], 
            # mujoco.mjtGeom.mjGEOM_CAPSULE, 
            # np.zeros(3), np.zeros(3), np.zeros(9), np.array([1, 0., 0., 1]))

            # # change setting : change value of geom
            # mujoco.mjv_connector(viewer.user_scn.geoms[viewer.user_scn.ngeom - 1], 
            #         mujoco.mjtGeom.mjGEOM_CAPSULE, 0.01,
            #         data.site('FFtip').xpos-0.001,
            #         data.site('FFtip').xpos)
        force_list.append(data.sensor("ft_sensor_force").data.copy())
        f_imp_list.append(F_imp)
        # data.ctrl[8:17] = q[8:17] * np.sin(data.time)*2
        #Step the simulation.
        mj.mj_step(model, data)

        x_error = np.subtract(goal, data.site('FFtip').xpos)
        viewer.sync()
    # else : 
    #     print("completed")

_, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 10))
force_list = np.array(force_list)
f_imp_list = np.array(f_imp_list)
sim_time = range(time)
lines = ax[0].plot(sim_time,force_list, label='Force sensor')
ax[0].set_title('Force sensor')
ax[0].set_ylabel('Newtons')
ax[0].set_xlabel('time')
ax[0].legend(iter(lines), ('$F_x$', '$F_y$', '$F_z$'))

lines = ax[1].plot(sim_time,f_imp_list, label='Impedance controller')
ax[1].set_title('Impedance controller')
ax[1].set_ylabel('Newtons')
ax[1].set_xlabel('time')
ax[1].legend(iter(lines), ('$F_x$', '$F_y$', '$F_z$', '$M_x$', '$M_y$', '$M_z$'))
plt.tight_layout()
plt.show()