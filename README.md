# Meadowrun DALL·E Mini + GLID3-XL + SwinIR pipeline demo

[This notebook](meadowrun-image-generation.ipynb) shows how to run DALL·E Mini (a
text-to-image model), GLID-3-xl (a diffusion model), and SwinIR (an upscaling model) in
your own AWS account via Meadowrun. This notebook chains together these models to
generate images.

Examples of images generated with this notebook of "a snake floating in clouds in the style of [van gogh|rene magritte|escher|gustav klimt]"
![Generated images of snakes floating through clouds](snakes-floating-through-clouds-composite.png)

To run this notebook:

```shell
# Clone this repo and create the local environment
git clone https://github.com/meadowdata/meadowrun-dallemini-demo
cd meadowrun-dallemini-demo
python3 -m venv venv
source venv/bin/activate
pip install -r local_requirements.txt

# Install meadowrun in your AWS account
meadowrun-manage-ec2 install --allow-authorize-ips
# Create an S3 bucket to cache pretrained models
aws s3 mb s3://meadowrun-dallemini
# Grant permission to Meadowrun to access this bucket
meadowrun-manage-ec2 grant-permission-to-s3-bucket meadowrun-dallemini

# Run a jupyter server
jupyter notebook
```

You'll also need to make sure your AWS account has non-zero quotas for at least some GPU instance types:
- L-3819A6DF: [All G and VT Spot Instance Requests](https://console.aws.amazon.com/servicequotas/home/services/ec2/quotas/L-3819A6DF)
- L-7212CCBC: [All P Spot Instance Requests](https://console.aws.amazon.com/servicequotas/home/services/ec2/quotas/L-7212CCBC)
- L-DB2E81BA: [Running On-Demand G and VT instances](https://console.aws.amazon.com/servicequotas/home/services/ec2/quotas/L-DB2E81BA)
- L-417A185B: [Running On-Demand P instances](https://console.aws.amazon.com/servicequotas/home/services/ec2/quotas/L-417A185B)
