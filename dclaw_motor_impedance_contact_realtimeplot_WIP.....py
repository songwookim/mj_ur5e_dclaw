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
opt = mj.MjvOption()                        # visualization options
spec = mj.MjSpec()    

#Reset state and time.
mujoco.mj_resetData(model, data)



tol = 0.01
dt = 0.0025
damping = 0.25
goal = [-0., -0.0, 0.1] #Desire position



#Init parameters
jacp = np.zeros((3,3, model.nv)) #translation jacobian (NUMBER OF JOINT x NUM_OF_ACTUATORS)
jacr = np.zeros((3,3, model.nv)) #rotational jacobian

#Simulate
desired_stiffness = 10
desired_damping = 3
desired_inertia = 8
#Get error.
end_effector_id = []
end_effector_id.append(model.body('FFL12').id)
end_effector_id.append(model.body('MFL22').id)
end_effector_id.append(model.body('THL32').id)

# current_pose = data.body(end_effector_id).xpos #Current pose
# x_error = np.subtract(goal, current_pose) #Init Error

# force_list = [0,0,0]
# f_imp_list = [0,0,0]
_, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 10))
force_list = np.zeros(3)
f_imp_list = np.zeros(9)

# -------------------------------------------
time = 0
lines1 = ax[0].plot(time,[force_list], label='Force sensor')
ax[0].set_title('Force sensor')
ax[0].set_ylabel('Newtons')
ax[0].set_xlabel('time')
ax[0].legend(iter(lines1), ('$F_x$', '$F_y$', '$F_z$'))

lines2 = ax[1].plot(time,[f_imp_list], label='Impedance controller')
ax[1].set_title('Impedance controller')
ax[1].set_ylabel('Newtons')
ax[1].set_xlabel('time')
ax[1].legend(iter(lines2), ('FFL10','FFL11','FFL12','MFL20','MFL21','MFL22','THL30','THL31','THL32'))
# -------------------------------------------
plt.tight_layout()
plt.draw()
tips = ["FFtip", "MFtip", "THtip"]
with mujoco.viewer.launch_passive(model, data) as viewer:
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
    
    # FFtip_pos1 = data.site("FFtip").xpos

    
    
    while viewer.is_running():
        F_imp = np.zeros([1,9])
        time += 1
        for idx, tip in enumerate(tips):
            x_error = np.subtract(goal, data.site(tip).xpos)
            
            mujoco.mj_jac(model, data, jacp[idx, :], jacr[idx, :], data.site(tip).xpos, end_effector_id[idx])
            xvel = jacp[idx, :]@data.qvel 
        # -------------------------------------------
            FFtip_pos1 = data.site(tip).xpos
            FFtip_pos2 = np.zeros(3)
            FFtip_vel1 = np.zeros(3)
            FFtip_vel2 = np.zeros(3)
            FFtip_acc = np.zeros(3)
            FFtip_pos2 = data.site(tip).xpos.copy()
            FFtip_vel2 = (FFtip_pos2 - FFtip_pos1) / dt
            FFtip_acc = (FFtip_vel2 - FFtip_vel1) / dt

            FFtip_vel1 = FFtip_vel2
            FFtip_pos1 = FFtip_pos2
        # -------------------------------------------

            xacc = FFtip_acc   
            F_imp += jacp[idx].T @ (desired_inertia*xacc + desired_damping*xvel + desired_stiffness*x_error) 
        # mj.mj_rnePostConstraint(model,data)
        data.qfrc_applied = F_imp

        mj.mj_forward(model, data)

        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
        
        # analyze the plot


        # force_list.append(data.sensor("ft_sensor_force").data.copy())
        # f_imp_list.append(F_imp[0])
        ax[0].plot(time,[data.sensor("ft_sensor_force").data.copy()], label='Force sensor')
        ax[1].plot(time,[F_imp[0]], label='Impedance controller')
        lines1[0]._x = time
        lines1._y = [data.sensor("ft_sensor_force").data.copy()]
        # lines1.set_ydata([data.sensor("ft_sensor_force").data.copy()])
        # plt.gca().lines[0].set_xdata([data.sensor("ft_sensor_force").data.copy()])
        #Step the simulation.
        mj.mj_step(model, data)

        viewer.sync()
    else : 
        print("completed")



pass