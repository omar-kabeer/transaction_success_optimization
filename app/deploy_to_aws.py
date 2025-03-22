import boto3
import time
import json

def deploy_to_ecs():
    """Deploy Docker container to AWS ECS"""
    
    # Initialize AWS clients
    ecr_client = boto3.client('ecr')
    ecs_client = boto3.client('ecs')
    
    # Get ECR repository URI
    repositories = ecr_client.describe_repositories(
        repositoryNames=['transaction-success-predictor']
    )
    repository_uri = repositories['repositories'][0]['repositoryUri']
    
    # Update ECS task definition
    task_definition = ecs_client.register_task_definition(
        family='transaction-predictor-task',
        networkMode='awsvpc',
        executionRoleArn='arn:aws:iam::123456789012:role/ecsTaskExecutionRole',
        containerDefinitions=[
            {
                'name': 'transaction-predictor',
                'image': f'{repository_uri}:latest',
                'essential': True,
                'portMappings': [
                    {
                        'containerPort': 8000,
                        'hostPort': 8000,
                        'protocol': 'tcp'
                    }
                ],
                'logConfiguration': {
                    'logDriver': 'awslogs',
                    'options': {
                        'awslogs-group': '/ecs/transaction-predictor',
                        'awslogs-region': 'us-east-1',
                        'awslogs-stream-prefix': 'ecs'
                    }
                },
                'healthCheck': {
                    'command': ['CMD-SHELL', 'curl -f http://localhost:8000/health || exit 1'],
                    'interval': 30,
                    'timeout': 5,
                    'retries': 3,
                    'startPeriod': 60
                },
                'memory': 512,
                'cpu': 256
            }
        ],
        requiresCompatibilities=['FARGATE'],
        cpu='256',
        memory='512'
    )
    
    # Update ECS service
    service = ecs_client.update_service(
        cluster='transaction-predictor-cluster',
        service='transaction-predictor-service',
        desiredCount=2,
        taskDefinition='transaction-predictor-task',
        forceNewDeployment=True
    )
    
    print(f"Deployment started. Task definition revision: {task_definition['taskDefinition']['revision']}")
    
    # Wait for deployment to complete
    waiter = ecs_client.get_waiter('services_stable')
    waiter.wait(
        cluster='transaction-predictor-cluster',
        services=['transaction-predictor-service']
    )
    
    print("Deployment completed successfully")

if __name__ == "__main__":
    deploy_to_ecs()