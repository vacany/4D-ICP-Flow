import random
import os
import time
import torch
import open3d as o3d
from tqdm import tqdm
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN

from sceneflow3d.datasets.argoverse2 import Argoverse2SceneFlow
from sceneflow3d.visuals.utils import *
from sceneflow3d.utils.boxes import MinimumBoundingBox


def estimateMinimumAreaBox(pts):
    Box = MinimumBoundingBox(pts[:,:2])

    l,w, theta = Box.length_parallel, Box.length_orthogonal, Box.unit_vector_angle
    h = np.max(pts[:,2]) - np.min(pts[:,2])

    center = Box.rectangle_center

    corner_pts = np.array(list(Box.corner_points))
    corner_pts = np.insert(corner_pts, 4, corner_pts[0], axis=0)
    corner_pts = corner_pts[[0,2,1,3,4]]
    box = np.array((center[0], center[1], h / 2 + np.min(pts[:,2]), l, w, h, theta))

    return box, corner_pts





def ICP4dSolver(time_pts_list, available_times, threshold=2.5, per_frame_move_thresh=0.05, RECONSTRUCT=False):
    
    trans_init = np.eye(4)
    boxes = []
    # corner_pts_list = []
    trans_list = []

    for t_idx, t in enumerate(sorted(available_times)):
        
        
        # if t != 0 and RECONSTRUCT:
            # print('not implemented correctly')
            # target_pts = np.concatenate(time_pts_list[:t_idx], axis=0)
        # else:
        target_pts = instance_pts[mask1, :3]
        box_t, corner_pts_t = estimateMinimumAreaBox(target_pts)
        target_pts = target_pts - box_t[:3]
        
        mask2 = instance_pts[:, -1] == t
        
        source_pts = instance_pts[mask2, :3]
        box_s, corner_pts_s = estimateMinimumAreaBox(source_pts)
        source_pts = source_pts - box_s[:3]    


        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(source_pts)

        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(target_pts)
            

        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        

        trans = reg_p2p.transformation
        trans_init = trans 

        traj_trans = trans.copy()    # inverse

        trans_list.append(traj_trans)
        boxes.append(box_s)

    
    # Trajectory
    trajectory = []
    for i in range(len(time_pts_list)):
        trajectory_p = np.linalg.inv(trans_list[i])[:3, -1] + boxes[i][:3]
        trajectory.append(trajectory_p)

    trajectory = np.stack(trajectory)

    position_difference = np.linalg.norm(trajectory[0] - trajectory[-1], axis=0)

    dynamic = position_difference > per_frame_move_thresh * len(trajectory)

    return trajectory, dynamic, trans_list, boxes, 


if __name__ == '__main__':
    RECONSTRUCT = False
    EPS = 0.4
    MAX_ADJACENT_TIMES = 4
    Z_SCALE = 0.3
    MIN_NBR_PTS = 100
    MIN_INCLUDED_TIMES = 5

    # cluster_ids = HDBSCAN(min_cluster_size=10, min_samples=5, cluster_selection_epsilon=0.5).fit_predict(numpy_pts)
    numpy_pts = np.load()
    numpy_pts[:, 2] *= Z_SCALE
    cluster_ids = DBSCAN(eps=EPS, min_samples=10).fit_predict(numpy_pts)
    cluster_ids += 1

    dynamic_mask = np.zeros(cluster_ids.shape[0], dtype=bool)
    logic_mask = np.zeros(cluster_ids.shape[0], dtype=bool)
    trajectories = {}
    boxes_dict = {}
    dynamic_id = {}

    for i in tqdm(range(cluster_ids.max())):
        if i == 0: continue # noise

        i_mask = cluster_ids == i
        instance_pts = numpy_pts[i_mask]

        if len(instance_pts) < MIN_NBR_PTS:
            continue
        
        if len(np.unique(instance_pts[:,-1])) < MIN_INCLUDED_TIMES:
            continue

        available_times = sorted(np.unique(instance_pts[:,-1]))
        mask1 = instance_pts[:, -1] == available_times[0]

        time_pts_list = [instance_pts[:, :4][instance_pts[:, -1] == t] for t in available_times]
        
        try:
            trajectory, dynamic, trans_list, boxes = ICP4dSolver(time_pts_list, available_times, threshold=3.5, per_frame_move_thresh=0.1, RECONSTRUCT=False)
            dynamic_mask[i_mask] = dynamic
            dynamic_id[i] = dynamic
            trajectories[i] = trajectory
            boxes_dict[i] = boxes
            logic_mask[i_mask] = True
            
        except:
            # Sometimes, there is too few points or skipped frames by occlussion.
            # These problems are not handled in this code.
            continue

    # Plotting
    plt.close()
    plt.figure(dpi=100, figsize=(10,10))
    plt.plot(numpy_pts[~dynamic_mask, 0], numpy_pts[~dynamic_mask, 1], 'b.', markersize=.3)

    for i in trajectories.keys():
        if dynamic_id[i]:
            numpy_pts_i = numpy_pts[cluster_ids == i]
            plt.plot(numpy_pts_i[:, 0], numpy_pts_i[:, 1], '.', markersize=.6)


    for i in trajectories.keys():
        # Plot only dynamic trajectories
        if dynamic_id[i]:
            plt.plot(trajectories[i][:,0], trajectories[i][:,1], marker='+', color='k', markersize=4, linestyle='-', linewidth=1.0)


    plt.tight_layout()
    plt.axis('equal')
    plt.title("PONE - Sequence")
    plt.savefig('assets/PONE_4DICP.png')
        
    for t_idx, t in enumerate(np.unique(numpy_pts[:,-1])):
        alpha = 1 - t / numpy_pts[:,-1].max()
        mask = numpy_pts[:,-1] <= t
        color_id = cluster_ids[mask]
        color_id *= dynamic_mask[mask]
        # color_id = dynamic_mask[mask]
        plt.close()
        plt.figure(dpi=150)
        plt.xlim(-60, 60)
        plt.ylim(-30, 30)
        plt.axis('equal')
        plt = plot_points(plt, numpy_pts[mask, :3], cluster_ids=color_id, markersize=1.)
        plt.title('PONE Sequence - Dynamic Instances')
        plt.savefig(f'assets/PONE_{t_idx:03d}.png')