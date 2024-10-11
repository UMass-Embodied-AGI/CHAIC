from mmdet.apis import init_detector, inference_detector

config_file = './detection_pipeline/config.py'
checkpoint_file = './detection_pipeline/epoch_4.pth'
model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'

print(inference_detector(model, 'detection_pipeline/example.png'))
# Visualize results