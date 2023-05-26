import numpy as np
import glob
import torch.utils.data
import torch
import os

class YcbTrain(torch.utils.data.Dataset):
    def __init__(self, root_dir, pcd_dir, scale_aug=False, rot_aug=False, npoint=1024, test_mode=False):
        self.classnames = ['black_and_decker_lithium_drill_driver']


        self.root_dir = root_dir
        self.pcd_dir = pcd_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        self.npoints = npoint
        set_ = root_dir.split('/')[-1]
        parent_dir = root_dir.rsplit('/', 2)[0]
        self.filepaths = []
        self.filepaths_gt = []
        self.array_files = []

        for i in range(len(self.classnames)):
            all_files = sorted(glob.glob(parent_dir + '/' + self.classnames[i] + '/' + set_ + '/*.xyz'))
            all_files_gt = sorted(glob.glob(parent_dir + '/' + self.classnames[i] + '/' + 'gt' + '/*.xyz'))

            self.filepaths.extend(all_files)
            self.filepaths_gt.extend(all_files_gt)



    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):

        path = self.filepaths[idx]
        class_name = path.split('/')[-3]
        p = os.path.split(path)[-1][:-4]
        p = p[:-1]
        p = p + 'y'
        gt = self.pcd_dir + '/' + class_name + '/train/' + p + '.xyz'


        class_id = self.classnames.index(class_name)
        partial = np.loadtxt(self.filepaths[idx])
        gt_dir = gt
        gt_load = np.loadtxt(gt_dir)
        return (partial, gt_load, class_id )


class YcbTest(torch.utils.data.Dataset):
    def __init__(self, root_dir, pcd_dir, scale_aug=False, rot_aug=False, npoint=1024, test_mode=False):
        self.classnames = ['black_and_decker_lithium_drill_driver',
                           'brine_mini_soccer_ball',
]

        self.root_dir = root_dir
        self.pcd_dir = pcd_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        self.npoints = npoint
        set_ = root_dir.split('/')[-1]
        parent_dir = root_dir.rsplit('/', 2)[0]
        self.filepaths = []
        self.array_files = []

        for i in range(len(self.classnames)):
            all_files = sorted(glob.glob(parent_dir + '/' + self.classnames[i] + '/' + set_ + '/*.xyz'))


            self.filepaths.extend(all_files)


    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
     path = self.filepaths[idx]
     class_name = path.split('/')[-3]
     p = os.path.split(path)[-1][:-4]
     p = p[:-1]
     p = p + 'y'
     gt = self.pcd_dir + '/' + class_name + '/test/' + p + '.xyz'


     class_id = self.classnames.index(class_name)
     partial = np.loadtxt(self.filepaths[idx])
     gt_dir = gt
     gt_load = np.loadtxt(gt_dir)
     return (partial, gt_load, class_name)


class YcbVal(torch.utils.data.Dataset):
 def __init__(self, root_dir, pcd_dir, scale_aug=False, rot_aug=False, npoint=1024, test_mode=False):
  self.classnames = ['black_and_decker_lithium_drill_driver',
                     'brine_mini_soccer_ball',
                     'campbells_condensed_tomato_soup',
                     'clorox_disinfecting_wipes_35',
                     'comet_lemon_fresh_bleach',
                     'domino_sugar_1lb',
                     'frenchs_classic_yellow_mustard_14oz',
                     'melissa_doug_farm_fresh_fruit_lemon',
                     'morton_salt_shaker',
                     'play_go_rainbow_stakin_cups_1_yellow',
                     'pringles_original',
                     'rubbermaid_ice_guard_pitcher_blue',
                     'soft_scrub_2lb_4oz',
                     'sponge_with_textured_cover',
                     'avocado_poisson_000',
                     'trash_can_poisson_003',
                     'remote_poisson_008',
                     'book_poisson_011',
                     'remote_poisson_004',
                     'avocado_poisson_000',
                     'camera_poisson_005',
                     'jar_poisson_008',
                     'trash_can_poisson_034',
                     'tetra_pak_poisson_012',
                     'mushroom_poisson_006',
                     'flashlight_poisson_013',
                     'champagne_glass_poisson_001',
                     'flashlight_poisson_000',
                     'trash_can_poisson_043',
                     'knife_poisson_019',
                     'notebook_poisson_021',
                     'remote_poisson_017',
                     'donut_poisson_008',
                     'hammer_poisson_008',
                     'violin_poisson_012',
                     'camera_poisson_006',
                     'tetra_pak_poisson_001',
                     'trash_can_poisson_046',
                     'box_poisson_008',
                     'wrench_poisson_018',
                     'jar_poisson_018',
                     'toy_poisson_015',
                     'hammer_poisson_005',
                     'camera_poisson_000',
                     'jar_poisson_000',
                     'wrench_poisson_022',
                     'stapler_poisson_004',
                     'pot_poisson_000',
                     'notebook_poisson_033',
                     'hammer_poisson_009',
                     'box_poisson_000',
                     'book_poisson_006',
                     'can_poisson_002',
                     'wrench_poisson_011',
                     'hammer_poisson_029',
                     'trash_can_poisson_024',
                     'boltcutter_poisson_000',
                     'book_poisson_019',
                     'cellphone_poisson_037',
                     'toothpaste_tube_poisson_003',
                     'pitcher_poisson_001',
                     'screwdriver_poisson_002',
                     'stapler_poisson_025',
                     'mushroom_poisson_013',
                     'box_poisson_022',
                     'hammer_poisson_012',
                     'dumpbell_poisson_001',
                     'orange_poisson_000',
                     'wrench_poisson_012',
                     'stapler_poisson_002',
                     'jar_poisson_011',
                     'book_poisson_009',
                     'egg_poisson_005',
                     'mug_new_poisson_001',
                     'stapler_poisson_003',
                     'cellphone_poisson_021',
                     'box_poisson_017',
                     'toy_poisson_001',
                     'cellphone_poisson_032',
                     'spray_bottle_poisson_000',
                     'screwdriver_poisson_022',
                     'light_bulb_poisson_001',
                     'trash_can_poisson_035',
                     'trash_can_poisson_049',
                     'box_poisson_002',
                     'watering_can_poisson_006',
                     'knife_poisson_004',
                     'box_poisson_023',
                     'hammer_poisson_031',
                     'bowl_poisson_005',
                     'toy_poisson_003',
                     'egg_poisson_003',
                     'stapler_poisson_023',
                     'wrench_poisson_013',
                     'bowl_poisson_021',
                     'bowling_pin_poisson_000',
                     'violin_poisson_008',
                     'toy_poisson_011',
                     'hammer_poisson_017',
                     'tape_poisson_005',
                     'book_poisson_002',
                     'remote_poisson_021',
                     'pitcher_poisson_004',
                     'toilet_paper_poisson_004',
                     'ladle_poisson_000',
                     'mushroom_poisson_010',
                     'banana_poisson_000',
                     'banana_poisson_004',
                     'wrench_poisson_014',
                     'jar_poisson_010',
                     'toaster_poisson_010',
                     'flashlight_poisson_015',
                     'camera_poisson_007',
                     'cellphone_poisson_011',
                     'knife_poisson_011',
                     'hammer_poisson_034',
                     'remote_poisson_018',
                     'knife_poisson_002',
                     'mushroom_poisson_011',
                     'boot_poisson_006',
                     'flashlight_poisson_010',
                     'camera_poisson_019',
                     'box_poisson_003',
                     'cellphone_poisson_017',
                     'saucepan_poisson_000',
                     'camera_poisson_017',
                     'violin_poisson_010',
                     'toy_poisson_006',
                     'knife_poisson_027',
                     'lemon_poisson_003',
                     'rubber_duck_poisson_001',
                     'stapler_poisson_022',
                     'trash_can_poisson_048',
                     'cellphone_poisson_008',
                     'tetra_pak_poisson_016',
                     'figurine_poisson_005',
                     'jar_poisson_013',
                     'toilet_paper_poisson_005',
                     'vase_poisson_004',
                     'wrench_poisson_009',
                     'cellphone_poisson_002',
                     'wrench_poisson_005',
                     'screwdriver_poisson_019',
                     'trash_can_poisson_039',
                     'toaster_poisson_000',
                     'stapler_poisson_000',
                     'cellphone_poisson_031',
                     'pot_poisson_003',
                     'cellphone_poisson_009',
                     'light_bulb_poisson_003',
                     'cellphone_poisson_012',
                     'trash_can_poisson_010',
                     'book_poisson_000',
                     'cellphone_poisson_005',
                     'donut_poisson_001',
                     'camera_poisson_002',
                     'screwdriver_poisson_021',
                     'detergent_bottle_poisson_003',
                     'box_poisson_001',
                     'flashlight_poisson_011',
                     'box_poisson_005',
                     'pliers_poisson_012',
                     'box_poisson_020',
                     'tetra_pak_poisson_003',
                     'rubik_cube_poisson_000',
                     'trash_can_poisson_022',
                     'light_bulb_poisson_006',
                     'pot_poisson_001',
                     'soccer_ball_poisson_003',
                     'cellphone_poisson_024',
                     'book_poisson_023',
                     'mushroom_poisson_004',
                     'egg_poisson_000',
                     'tetra_pak_poisson_000',
                     'cellphone_poisson_028',
                     'cellphone_poisson_039',
                     'banjo_poisson_001',
                     'bottle_new_poisson_000',
                     'spatula_poisson_001',
                     'light_bulb_poisson_009',
                     'pitcher_poisson_003',
                     'rubik_cube_poisson_004',
                     'jar_poisson_020',
                     'tetra_pak_poisson_015',
                     'wrench_poisson_002',
                     'pitcher_poisson_005',
                     'violin_poisson_015',
                     'screwdriver_poisson_000',
                     'pliers_poisson_015',
                     'light_bulb_poisson_007',
                     'jar_poisson_006',
                     'camera_poisson_015',
                     'cellphone_poisson_033',
                     'cellphone_poisson_003',
                     'kettle_poisson_004',
                     'can_poisson_008',
                     'cellphone_poisson_014',
                     'banana_poisson_002',
                     'mushroom_poisson_008',
                     'spatula_poisson_002',
                     'tomato_poisson_000',
                     'cellphone_poisson_034',
                     'notebook_poisson_035',
                     'remote_poisson_014',
                     'notebook_poisson_053',
                     'toaster_poisson_005',
                     'tetra_pak_poisson_013',
                     'egg_poisson_012',
                     'egg_poisson_001',
                     'avocado_poisson_001',
                     'box_poisson_014',
                     'jar_poisson_015',
                     'remote_poisson_022',
                     'lime_poisson_001',
                     'remote_poisson_000',
                     'tape_poisson_004',
                     'shampoo_poisson_001',
                     'can_poisson_018',
                     'bowl_poisson_028',
                     'camera_poisson_003',
                     'tetra_pak_poisson_006',
                     'trash_can_poisson_045',
                     'stapler_poisson_010',
                     'donut_poisson_006',
                     'camera_poisson_020',
                     'knife_poisson_030',
                     'box_poisson_016',
                     'jar_poisson_012',
                     'lemon_poisson_002',
                     'tetra_pak_poisson_025',
                     'can_poisson_015',
                     'light_bulb_poisson_004',
                     'remote_poisson_006',
                     'box_poisson_019',
                     'hammer_poisson_035',
                     'tetra_pak_poisson_020',
                     'sweet_corn_poisson_000',
                     'tetra_pak_poisson_008',
                     'donut_poisson_005',
                     'tetra_pak_poisson_022',
                     'jar_poisson_001',
                     'book_poisson_022',
                     'box_new_poisson_000',
                     'box_poisson_021',
                     'stapler_poisson_007',
                     'toilet_paper_poisson_006',
                     'notebook_poisson_043',
                     'cellphone_poisson_025',
                     'soccer_ball_poisson_000',
                     'jar_poisson_003',
                     'wrench_poisson_008',
                     'box_poisson_006',
                     'jar_poisson_024',
                     'jar_poisson_021',
                     'mushroom_poisson_012',
                     'pitcher_poisson_000',
                     'cellphone_poisson_035',
                     'tetra_pak_poisson_014',
                     'remote_poisson_010',
                     'mushroom_poisson_007',
                     'screwdriver_poisson_006',
                     'lemon_poisson_004',
                     'tetra_pak_poisson_018',
                     'cellphone_poisson_016',
                     'boot_poisson_004',
                     'jar_poisson_007',
                     'hammer_poisson_027',
                     'cellphone_poisson_013',
                     'can_poisson_006',
                     'toaster_poisson_007',
                     'cellphone_poisson_020',
                     'pliers_poisson_000',
                     'tetra_pak_poisson_011',
                     'tetra_pak_poisson_004',
                     'book_poisson_008',
                     'watering_can_poisson_003',
                     'can_poisson_007',
                     'stapler_poisson_008',
                     'screwdriver_poisson_024',
                     'binder_poisson_004',
                     'camera_poisson_013',
                     'hammer_poisson_025',
                     'pliers_poisson_002',
                     'shampoo_new_poisson_000',
                     'detergent_bottle_poisson_004',
                     'knife_poisson_031',
                     'remote_poisson_012',
                     'remote_poisson_009',
                     'cellphone_poisson_004',
                     'notebook_poisson_009',
                     'hammer_poisson_004',
                     'remote_poisson_016',
                     'remote_poisson_020',
                     'spray_bottle_poisson_003',
                     'trash_can_poisson_008',
                     'flashlight_poisson_003',
                     'light_bulb_poisson_000',
                     'trash_can_poisson_037',
                     'starfruit_poisson_002',
                     'can_poisson_005',
                     'detergent_bottle_poisson_001',
                     'donut_poisson_002',
                     'box_poisson_013',
                     'trash_can_poisson_047',
                     'jar_poisson_014',
                     'flashlight_poisson_014',
                     'watermelon_poisson_000',
                     'box_poisson_011',
                     'egg_poisson_007',
                     'toy_poisson_008',
                     'toy_poisson_005',
                     'trash_can_poisson_011',
                     'watering_can_poisson_000',
                     'flashlight_poisson_002',
                     'hammer_poisson_002',
                     'cellphone_poisson_036',
                     'can_poisson_009',
                     'can_poisson_010',
                     'bowl_poisson_030',
                     'zucchini_poisson_001',
                     'toilet_paper_poisson_001',
                     'pliers_poisson_004',
                     'hammer_poisson_020',
                     'tape_poisson_003',
                     'donut_poisson_000',
                     'hammer_poisson_011',
                     'spray_can_poisson_005',
                     'stapler_poisson_026',
                     'jar_poisson_002',
                     'pliers_poisson_001',
                     'trash_can_poisson_007',
                     'banana_poisson_003',
                     'stapler_poisson_028',
                     'notebook_poisson_037',
                     'notebook_poisson_051',
                     'can_poisson_017',
                     'toothpaste_tube_poisson_002',
                     'watering_can_poisson_001',
                     'book_poisson_004',
                     'violin_poisson_016',
                     'kettle_poisson_007',
                     'soccer_ball_poisson_006',
                     'spray_can_poisson_003',
                     'spray_can_poisson_001',
                     'screwdriver_poisson_005',
                     'detergent_bottle_poisson_007',
                     'camera_poisson_014',
                     'jar_poisson_022',
                     'bowl_poisson_016',
                     'can_poisson_021',
                     'trash_can_poisson_040',
                     'toy_poisson_019',
                     'knife_poisson_021',
                     'wrench_poisson_006',
                     'wrench_poisson_017',
                     'trash_can_poisson_014',
                     'box_poisson_026',
                     'screwdriver_poisson_003',
                     'flashlight_poisson_009',
                     'detergent_bottle_poisson_002',
                     'bowl_poisson_015',
                     'trash_can_poisson_033',
                     'trash_can_poisson_031',
                     'soccer_ball_poisson_005',
                     'toy_poisson_025',
                     'light_bulb_poisson_005',
                     'lute_poisson_000',
                     'trash_can_poisson_021',
                     'tennis_ball_poisson_000',
                     'can_poisson_024',
                     'toilet_paper_poisson_003',
                     'light_bulb_poisson_002',
                     'mushroom_poisson_005',
                     'cellphone_poisson_027',
                     'mushroom_poisson_014',
                     'banana_poisson_005',
                     'remote_poisson_005',
                     'cellphone_poisson_029',
                     'screwdriver_poisson_023',
                     'rubber_duck_poisson_000',
                     'flashlight_poisson_005',
                     'orange_poisson_001',
                     'trash_can_poisson_005',
                     'soccer_ball_poisson_004',
                     'trash_can_poisson_044',
                     'light_bulb_poisson_008',
                     'flashlight_poisson_006',
                     'wrench_poisson_016',
                     'bowl_poisson_025',
                     'can_poisson_022',
                     'donut_poisson_003',
                     'can_poisson_000',
                     'trash_can_poisson_025',
                     'stapler_poisson_012',
                     'jar_poisson_005',
                     'trash_can_poisson_041',
                     'violin_poisson_013',
                     'screwdriver_poisson_013',
                     'pliers_poisson_006',
                     'notebook_poisson_020',
                     'spray_can_poisson_000',
                     'wrench_poisson_007',
                     'can_poisson_014',
                     'screwdriver_poisson_008',
                     'lemon_poisson_001',
                     'spray_bottle_poisson_002',
                     'spray_can_poisson_002',
                     'screwdriver_poisson_009',
                     'camera_poisson_012',
                     'hammer_poisson_023',
                     'camera_poisson_010',
                     'egg_poisson_006',
                     'banjo_poisson_002',
                     'notebook_poisson_041',
                     'jar_poisson_004',
                     'hammer_poisson_018',
                     'trash_can_poisson_026',
                     'shampoo_new_poisson_001',
                     'shovel_poisson_000',
                     'kettle_poisson_006',
                     'trash_can_poisson_023',
                     'knife_poisson_001',
                     'pot_poisson_004',
                     'knife_poisson_025',
                     'toaster_poisson_006',
                     'tape_poisson_000',
                     'pliers_poisson_016',
                     'remote_poisson_003',
                     'pliers_poisson_014',
                     'toy_poisson_004',
                     'screwdriver_poisson_011',
                     'donut_poisson_004',
                     'egg_poisson_002',
                     'hammer_poisson_006',
                     'screwdriver_poisson_004',
                     'tetra_pak_poisson_009',
                     'camera_poisson_018',
                     'tetra_pak_poisson_005',
                     'binder_poisson_000',
                     'pliers_poisson_017',
                     'violin_poisson_019']

  self.root_dir = root_dir
  self.pcd_dir = pcd_dir
  self.scale_aug = scale_aug
  self.rot_aug = rot_aug
  self.test_mode = test_mode
  self.npoints = npoint
  set_ = root_dir.split('/')[-1]
  parent_dir = root_dir.rsplit('/', 2)[0]
  self.filepaths = []
  self.array_files = []

  for i in range(len(self.classnames)):
   all_files = sorted(glob.glob(parent_dir + '/' + self.classnames[i] + '/' + set_ + '/*.xyz'))

   self.filepaths.extend(all_files)

 def __len__(self):
  return len(self.filepaths)

 def __getitem__(self, idx):
  path = self.filepaths[idx]
  class_name = path.split('/')[-3]
  p = os.path.split(path)[-1][:-4]
  p = p[:-1]
  p = p + 'y'
  gt = self.pcd_dir + '/' + class_name + '/test/' + p + '.xyz'
  # class_name = class_name  + '/train_models_holdout_views'

  # gt = os.path.split(path)[0][:-5]
  # gt = gt + 'gt/'
  # dataname = os.path.split(path)[1]
  # dataname = dataname[:-5]
  # dataname = dataname + 'y.xyz'
  # gt = gt + dataname

  class_id = self.classnames.index(class_name)
  partial = np.loadtxt(self.filepaths[idx])
  gt_dir = gt
  gt_load = np.loadtxt(gt_dir)
  return (partial, gt_load, class_id)