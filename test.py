import easyocr
from easyocr import Reader
import os

# Set GPU devices
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


# Initialize both models
print("Loading both models...")
#reader_pretrained = Reader(['en', 'ch_tra'], gpu=True)
#reader_custom = Reader(['en', 'ch_tra'], gpu=True, model_storage_directory='./models', download_enabled=False)

reader_pretrained = Reader(['en', 'ch_tra'], gpu=True)
reader_custom = Reader(['en', 'ch_tra'], gpu=True, recog_network='custom', model_storage_directory='./user_network_dir',
                       user_network_directory='./user_network_dir')

# reader_pretrained = Reader(['ch_tra'], gpu=True)
# reader_custom = Reader(['ch_tra'], gpu=True, recog_network='custom', model_storage_directory='./user_network_dir',
#                        user_network_directory='./user_network_dir', download_enabled=False)


# # --- Define the arguments for your custom.py Model class ---
# # These parameters will be passed to your Model's __init__ method.
# # Based on your previous error, nh (hidden_size) must be 256.
# custom_model_params = {
#     'num_class': 8125,
#     'nc': 1,
#     'nh': 256,  # This is the critical parameter to fix the size mismatch
#     'output_channel': 512,
# }
# # -----------------------------------------------------------

# print("Loading custom model...")

# reader_custom = Reader(
#     ['en', 'ch_tra'],
#     gpu=True,

# # 1. Re-add this to explicitly name the network
#     recog_network='custom',

#     # 2. Use the older keyword to pass the arguments (it might only be allowed
#     #    if 'recog_network' is set to 'custom' in this version)
#     network_params=custom_model_params,

#     # 3. Specify the location of custom.py and custom.pth
#     user_network_directory='./user_network_dir',
#     model_storage_directory='./user_network_dir',

#     download_enabled=False
# )

print("Custom model loaded successfully!")

# These are the correct parameters for your saved model,
# deduced from your config and the error message.
# custom_model_params = {
#     'Transformation': 'None',
#     'FeatureExtraction': 'ResNet',
#     'SequenceModeling': 'BiLSTM',
#     'Prediction': 'CTC',
#     'input_channel': 1,
#     'output_channel': 512,
#     # This value is from your error message (torch.Size([1024, ...]))
#     # which implies a hidden_size of 512, NOT 256.
#     'hidden_size': 512,
#     'num_class': 8125  # From your config
# }

# print("Loading custom model...")
# # Now, pass this dictionary to the Reader
# reader_custom = Reader(
#     ['en', 'ch_tra'],
#     gpu=True,
#     recog_network='custom',
#     network_params=custom_model_params,  # <--- THE FIX
#     model_storage_directory='./user_network_dir',
#     user_network_directory='./user_network_dir',
#     download_enabled=False
# )

print("Custom model loaded successfully!")

# reader_custom = Reader(['en', 'ch_tra'], gpu=True,
#                       model_storage_directory='./user_network_dir',
#                       user_network_directory='./user_network_dir',
#                       recog_network='custom')

# Get test images
demo_images_dir = './demo_images'
test_files = [f for f in os.listdir(demo_images_dir) if not f.startswith('.')]
test_files.sort()

print(f"\n{'='*80}")
print("SIDE-BY-SIDE COMPARISON: Pre-trained vs Custom Model")
print(f"{'='*80}\n")

# Compare results for each image
for filename in test_files:
    file_path = os.path.join(demo_images_dir, filename)
    print(f"📷 Testing: {filename}")
    print(f"{'-'*80}")

    # Test with pre-trained model
    results_pretrained = reader_pretrained.readtext(file_path)
    print("🔵 EasyOCR Pre-trained Model:")
    if results_pretrained:
        for bbox, text, confidence in results_pretrained:
            print("   confidence: %.4f, string: '%s'" % (confidence, text))
    else:
        print("   No text detected")

    print()

    # Test with custom model
    results_custom = reader_custom.readtext(file_path)
    print("🟢 Your Custom Model:")
    if results_custom:
        for bbox, text, confidence in results_custom:
            print("   confidence: %.4f, string: '%s'" % (confidence, text))
    else:
        print("   No text detected")

    print(f"\n{'='*80}\n")
