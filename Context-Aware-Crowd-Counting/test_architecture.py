import torch
from model import CANNet, CANNet_GRL_Frontend, CANNet_GRL_Context, CANNet_GRL_Concat
import dataset
import sys

def test_models():
    print("--------------------------------------------------")
    print("Testing Model Architectures with Dummy Input")
    print("--------------------------------------------------")
    # Batch = 2, Channels = 3, HxW = 256x256
    dummy_input = torch.randn(2, 3, 256, 256)
    
    print("Testing baseline CANNet...")
    model = CANNet()
    out = model(dummy_input)
    # the output depends on maxpooling, typically 1/8 so 32x32
    assert out.shape == (2, 1, 32, 32), f"Expected (2, 1, 32, 32), got {out.shape}"
    print(f"CANNet Output shape: {out.shape} -> PASS\n")
    
    models_to_test = [
        ("CANNet_GRL_Frontend", CANNet_GRL_Frontend(num_domains=12)),
        ("CANNet_GRL_Context", CANNet_GRL_Context(num_domains=12)),
        ("CANNet_GRL_Concat", CANNet_GRL_Concat(num_domains=12))
    ]
    
    for name, model in models_to_test:
        print(f"Testing {name}...")
        density, domain = model(dummy_input, alpha=0.5)
        
        assert density.shape == (2, 1, 32, 32), f"Expected density (2, 1, 32, 32), got {density.shape}"
        assert domain.shape == (2, 12), f"Expected domain (2, 12), got {domain.shape}"
        
        print(f"{name} Density shape: {density.shape} -> PASS")
        print(f"{name} Domain shape:  {domain.shape} -> PASS\n")

def test_dataset():
    print("--------------------------------------------------")
    print("Testing Dataset Domain Logic")
    print("--------------------------------------------------")
    img_to_country, num_domains = dataset._load_domain_data()
    print(f"Found {num_domains} unique domains (countries).")
    print(f"Mapped {len(img_to_country)} images from CSVs to corresponding domains.")
    
    if len(img_to_country) > 0:
        sample_items = list(img_to_country.items())[:3]
        print(f"Sample mappings (image filename -> country ID):")
        for filename, cid in sample_items:
            print(f"  {filename} -> {cid}")
    else:
        print("WARNING: Could not parse domains. Check metadata.csv or train.csv paths.")

if __name__ == "__main__":
    try:
        test_dataset()
        print("\n")
        test_models()
        print("All tests passed successfully!")
    except Exception as e:
        print(f"Test failed with error: {e}")
        sys.exit(1)
