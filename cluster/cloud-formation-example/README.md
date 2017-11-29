# TensorFlow


Create Stack:
```
aws --region ap-southeast-2 cloudformation create-stack --stack-name tensorflow --template-body file://tensorflow.yaml --parameters ParameterKey=KeyName,ParameterValue=[KeyName]
```

Update Stack:
```
aws --region ap-southeast-2 cloudformation update-stack --stack-name tensorflow --template-body file://tensorflow.yaml --parameters ParameterKey=KeyName,ParameterValue=[KeyName]
```

Delete Stack:
```
aws --region ap-southeast-2 cloudformation delete-stack --stack-name tensorflow
```

Describe Stack:
```
aws --region ap-southeast-2 cloudformation describe-stacks --stack-name tensorflow
```

# Create DNS zone distributed.tensorflow.
bash -x zone.sh create distributed.tensorflow. ap-southeast-2 vpc-9e314bfa
# Launch cluster with CloudFormation
aws --region ap-southeast-2 cloudformation create-stack --stack-name tensorflow --template-body file://tensorflow.yaml --parameters ParameterKey=KeyName,ParameterValue=ytang ParameterKey=SubnetId,ParameterValue=subnet-8eaba9ea ParameterKey=VPC,ParameterValue=vpc-9e314bfa
# Destroy cluster with CloudFormation
aws --region ap-southeast-2 cloudformation delete-stack --stack-name tensorflow
# Delete DNS zone distributed.tensorflow.
bash -x zone.sh delete distributed.tensorflow. ap-southeast-2 vpc-9e314bfa
