import mujoco as mj
import mujoco.viewer
from robot_descriptions import ur5e_mj_description
from robot_descriptions.loaders.mujoco import load_robot_description
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

model = mj.MjModel.from_xml_path("./universal_robots_ur5e_with_dclaw/dclaw_motor/dclaw3xh.xml")
matplotlib.use('TkAgg',force=True)
data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera() 
mujoco.mjv_defaultFreeCamera(model, cam)
opt = mj.MjvOption()                        # visualization options
spec = mj.MjSpec()    

#Reset state and time.
mujoco.mj_resetData(model, data)



tol = 0.01
dt = 0.0025
damping = 0.25
# goal = [-0., -0.0, 0.1] #Desire position
# goals = [[0., .0, 0.1],[0., -0.1, 0.1],[0., 0.1, 0.1]] #Desire position 1
goals = [[0., .0, 0.1],[0., .0, 0.1],[0., .0, 0.1]] #Desire position 2
# goal = [-0., -0.02, 0.1] #Desire position



#Init parameters
jacp = np.zeros((3,3, model.nv)) #translation jacobian (NUMBER OF JOINT x NUM_OF_ACTUATORS)
jacr = np.zeros((3,3, model.nv)) #rotational jacobian

#Simulate
desired_stiffness = [[500, 0, 0],[0, 500, 0],[0, 0, 500]]
desired_damping = [[1, 0, 0],[0, 1, 0],[0, 0, 1]]
# desired_damping = 300
desired_inertia = np.eye(3) * 0.01 # compare 1 with 30
#Get error.
end_effector_id = []
end_effector_id.append(model.body('FFL12').id)
end_effector_id.append(model.body('MFL22').id)
end_effector_id.append(model.body('THL32').id)

# current_pose = data.body(end_effector_id).xpos #Current pose
# x_error = np.subtract(goal, current_pose) #Init Error

force_list = []
f_imp_list = []
e_list = []
tips = ["FFtip", "MFtip", "THtip"]    # 
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.sync()
    with viewer.lock():
        for goal in goals:
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
        
    time = 0
    Tip_pos1 = np.array([data.site("FFtip").xpos, data.site("MFtip").xpos, data.site("THtip").xpos])
    Tip_pos2 = np.zeros([3,3])
    Tip_vel1 = np.zeros([3,3])
    Tip_vel2 = np.zeros([3,3])
    Tip_acc = np.zeros([3,3])
    Jacp_1 = jacp.copy()
    # Jacp_2 = np.zeros([3,3])
    while viewer.is_running():
        F_imp = np.zeros([1,9])
        time += 1
        tau_imp = 0
        e_temp = []
        for idx, tip in enumerate(tips):
            x_error = np.subtract(data.site(tip).xpos, goals[idx]) 
            
            mujoco.mj_jac(model, data, jacp[idx, :], jacr[idx, :], data.site(tip).xpos, end_effector_id[idx])
            n = jacp[idx, :].shape[1]
            I = np.identity(n)
            product = jacp[idx, :].T @ jacp[idx, :] + damping * I

            if np.isclose(np.linalg.det(product), 0):
                j_inv = np.linalg.pinv(product) @ jacp[idx, :].T
            else:
                j_inv = np.linalg.inv(product) @ jacp[idx, :].T
            

            xvel = jacp[idx, :]@data.qvel 
        # -------------------------------------------
            Jacp_2 = jacp.copy()
            Jacp_dot = (Jacp_2[idx, :] - Jacp_1[idx, :]) / dt
            Jacp_1 = Jacp_2.copy()
        # -------------------------------------------

        
            M = np.zeros((model.nv, model.nv)) # 9x9
            mj.mj_fullM(model, M, data.qM)

            c = data.qfrc_bias # bais force (centrifugal, coriolis, gravity)
            # c[3*len(tips):] = 0

            Md_inv = np.linalg.inv(desired_inertia) 

            f_ext = data.sensor("ft_sensor_force1").data.copy() * 0

            tau_imp += M@j_inv@(Md_inv@(f_ext+desired_damping@xvel-desired_stiffness@x_error)-Jacp_dot@data.qvel)+c-jacp[idx,:].T@f_ext
            e_temp.append(x_error)
        data.qfrc_applied  = tau_imp

        mj.mj_forward(model, data)

        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1

        f1 = np.sum(data.sensor("ft_sensor_force1").data.copy())
        f2 = np.sum(data.sensor("ft_sensor_force2").data.copy())
        f3 = np.sum(data.sensor("ft_sensor_force3").data.copy())

        e = np.array(e_temp)**2
        e1 = np.sum(e[0])
        e2 = np.sum(e[1])
        e3 = np.sum(e[2])
 
        force_list.append([f1,f2,f3])
        f_imp_list.append(tau_imp[0:3])
        e_list.append([e1,e2,e3])
        #Step the simulation.
        mj.mj_step(model, data)
        viewer.sync()
        if time > 5000:
            break
    # else : 
    #     print("completed")

_, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 10))
force_list = np.array(force_list)
f_imp_list = np.array(f_imp_list)
e_list = np.array(e_list)
sim_time = range(time)
lines = ax[0].plot(sim_time,force_list, label='Force sensor')
ax[0].set_title('Measured Force by Force sensor')
ax[0].set_ylabel('Newtons')
ax[0].set_xlabel('time(ms)')
ax[0].legend(iter(lines), ('$Finger 1$', '$Finger 2$', '$Finger 3$'))
# ax[0].set_xlim([-2, 2])
# ax[0].set_ylim([-2, 2])

lines = ax[1].plot(sim_time,e_list, label='Position error')
ax[1].set_title('Position error for each end-effectors ')
ax[1].set_ylabel('Position error')
ax[1].set_xlabel('time(ms)')
# ax[1].legend(iter(lines), ('FFL10','FFL11','FFL12','MFL20','MFL21','MFL22','THL30','THL31','THL32'))
ax[1].legend(iter(lines), ('Finger 1','Finger 2','Finger 3'))
# ax[1].set_xlim([-1, 1])
# ax[1].set_ylim([-1, 1])
plt.tight_layout()
plt.show()
plt.close()

# lines = ax[1].plot(sim_time,f_imp_list, label='Impedance controller')
# ax[1].set_title('$\\tau_{imp}$ by Impedance controller without Force Sensor')
# ax[1].set_ylabel('Newtons')
# ax[1].set_xlabel('time(ms)')
# # ax[1].legend(iter(lines), ('FFL10','FFL11','FFL12','MFL20','MFL21','MFL22','THL30','THL31','THL32'))
# ax[1].legend(iter(lines), ('Top','Middle','Bottom'))
# # ax[1].set_xlim([-1, 1])
# # ax[1].set_ylim([-1, 1])
# plt.tight_layout()
# plt.show()
# plt.close()
