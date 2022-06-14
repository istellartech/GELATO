import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def display_6DoF(out, flag_savefig=False):
    
    plt.figure()
    plt.title("Altitude[km]")
        
    plt.plot(out["time"], out["altitude"], ".-", lw=0.8, label="Altitude")
    plt.plot(out["time"], out["altitude_apogee"], label="altitude_apogee")
    plt.plot(out["time"], out["altitude_perigee"], label="altitude_perigee")

    plt.ylim([0,None])
    plt.xlim([0,None])
    plt.grid()
    plt.legend()
    if flag_savefig:
        plt.savefig("figures/Altitude.png")
    plt.show()

    plt.figure()
    plt.title("Orbital elements")
    plt.plot(out["time"], out.loc[:,["inclination","lon_ascending_node","argument_perigee"]], label=["i", "Ω", "ω"])
    plt.xlim([0,None])
    plt.ylim([-180,180])
    plt.grid()
    if flag_savefig:
        plt.savefig("figures/Orbital_Elements.png")
    plt.show()
    
    
    plt.figure()
    plt.title("Ground speed_NED")
    
    plt.plot(out["time"], out.loc[:,["vel_ground_NED_X","vel_ground_NED_Y","vel_ground_NED_Z"]], ".-", lw=0.8, label=["N", "E", "D"])
    plt.xlim([0,None])
    plt.grid()
    if flag_savefig:
        plt.savefig("figures/Ground_Speed.png")
    plt.show()

    plt.figure()
    plt.title("Angle of attack")
    
    plt.plot(out["time"], out.loc[:,["AOA_pitch_NED2BODY","AOA_yaw"]], ".-", lw=0.8, label=["pitch_NED2BODY", "yaw"])
    plt.xlim([0,None])
    plt.ylim([-10,10])
    plt.grid()
    if flag_savefig:
        plt.savefig("figures/Angle_of_attack.png")
    plt.show()
    
    plt.figure()
    plt.title("Flight trajectory and velocity vector")
    i = int(len(out["time"]) / 10)
    
    pos_llh = out.loc[:,["lat","lon","altitude"]].to_numpy("f8")
    vel_ned = out.loc[:,["vel_ground_NED_X","vel_ground_NED_Y","vel_ground_NED_Z"]].to_numpy("f8")

    x = np.deg2rad(pos_llh[:,1])
    y = np.log(np.tan(np.deg2rad(45+pos_llh[:,0]/2)))
    
    plt.plot(x,y,lw=0.5)
    plt.quiver(x[::i],y[::i],vel_ned[::i,1],vel_ned[::i,0], scale=1.5e5)
    plt.axis("equal")
    plt.grid()
    if flag_savefig:
        plt.savefig("figures/Trajectory.png")
    plt.show()


    plt.figure()
    plt.title("Thrust vector (ECI)")
    
    plt.plot(out["time"], out.loc[:,["thrust_direction_ECI_X","thrust_direction_ECI_Y","thrust_direction_ECI_Z"]], ".-", lw=0.8)
    plt.ylim([-1.0,1.0])
    plt.xlim([0, None])
    plt.grid()
    if flag_savefig:
        plt.savefig("figures/Thrust_vector.png")
    plt.show()
    
    
    plt.figure()
    plt.title("Euler Angle and Velocity Direction")
    
    
    plt.plot(out["time"], out.loc[:,["heading_NED2BODY","pitch_NED2BODY","roll_NED2BODY"]], ".-", lw=0.8, label=["heading", "pitch", "roll"])
    plt.plot(out["time"], out["azimuth_vel_inertial_geocentric"], lw=1, label="azimuth_vel_inertial")
    plt.plot(out["time"], out["flightpath_vel_inertial_geocentric"], lw=1, label="flightpath_vel_inertial")
    plt.xlim([0,None])
    plt.ylim([-180,180])
    plt.grid()
    plt.legend()
    if flag_savefig:
        plt.savefig("figures/Euler_angle.png")
    plt.show()
    
def display_3d(out):
    lim = 6378 + 2500

    x_km = out["pos_ECI_X"].to_numpy() / 1000.0
    y_km = out["pos_ECI_Y"].to_numpy() / 1000.0
    z_km = out["pos_ECI_Z"].to_numpy() / 1000.0

    thetas = np.linspace(0, np.pi, 20)
    phis = np.linspace(0, np.pi*2, 20)

    xs = 6378 * np.outer(np.sin(thetas),np.sin(phis))
    ys = 6378 * np.outer(np.sin(thetas),np.cos(phis))
    zs = 6357 * np.outer(np.cos(thetas),np.ones_like(phis))

    plt.figure(figsize=(8,8))
    ax = plt.axes(projection="3d")
    ax.set_box_aspect((1,1,1))

    ax.view_init(elev=15, azim=150)

    ax.plot_wireframe(xs,ys,zs, color="c", lw=0.2)
    ax.plot(x_km,y_km,z_km, color="r")

    ax.plot([0,2000],[0,0],[0,0],color="r",lw=1)
    ax.plot([0,0],[0,2000],[0,0],color="g",lw=1)
    ax.plot([0,0],[0,0],[0,2000],color="b",lw=1)


    ax.set_xlabel("X[km]")
    ax.set_ylabel("Y[km]")
    ax.set_zlabel("Z[km]")