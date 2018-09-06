import os

common_datasets_path = '/media/svakhreev/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/test'

config = {
    'categories': ['recumbent_person', 'person'],
    'infer_model_path': '../snapshots/current_person_recum/yolo3_model.h5',
    'verifiers': ['sergey'],
    'datasets': {
        'train': [
            {
                'path': 'recumbent_people_train',
                'images_count': 'all',
                'only_verified': True,
                'bbox_limits': [0, 0, 100, 100],
                'recursive': False
            }
        ],
        'val': [
            {
                'path': 'recumbent_people_val',
                'images_count': 'all',
                'only_verified': True,
                'bbox_limits': [0, 0, 100, 100],
                'recursive': False
            }
        ]
    },
    'train': {
        'net_size': 608,
        'max_roi_count': 100,
        'batch_size': 8,
        'epochs': 1000,
        'cuda_devices': '0',
        'classifier_weights_path': None
    },
    'eval': {
        'net_size': 416,
        'max_roi_count': 50,
        'batch_size': 1,
        'cuda_devices': '1',
        'classifier_weights_path': './classifier.h5'
    },
    'predict': {
        'net_size': 416,
        'confidence': 0.9,
        'nms_threshold': 0.4,
        'every_nth_frame': 25,
        'cuda_devices': '1',
        'classifier_weights_path': './classifier.h5',
        'videofile_path': '/mnt/nfs/Videos/Лежачие люди/лежачие люди видео/Worlds First Planking Flash Mob on YT.mp4'
    }
}


for mode in config['datasets'].keys():
    for folder in config['datasets'][mode]:
        assert os.path.exists(os.path.join(common_datasets_path, folder['path']))
        folder['path'] = os.path.join(common_datasets_path, folder['path'])
