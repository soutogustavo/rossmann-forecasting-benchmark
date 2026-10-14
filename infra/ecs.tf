variable "image_tag" {
  type        = string
  description = "ECS image tag for pipeline"
}

resource "aws_ecs_cluster" "main" {
  name = "gs-${local.environment}-rossmann"
}

resource "aws_cloudwatch_log_group" "pipeline" {
  name              = "/ecs/gs-${local.environment}-rossmann-pipeline"
  retention_in_days = 30
}

resource "aws_ecs_task_definition" "pipeline" {
  family                   = "gs-${local.environment}-rossmann-pipeline"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 2048 # 2 vCPU
  memory                   = 8192 # 8 GB
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  container_definitions = jsonencode([{
    name      = "pipeline"
    image     = "${aws_ecr_repository.pipeline.repository_url}:${var.image_tag}"
    essential = true
    command   = ["src.pipeline.weekly_pipeline", "--env", local.environment]
    environment = [
      { name = "MLFLOW_TRACKING_URI", value = data.tfe_outputs.platform.nonsensitive_values.mlflow_tracking_uri },
      { name = "AWS_DEFAULT_REGION", value = "eu-central-1" },
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.pipeline.name
        awslogs-region        = "eu-central-1"
        awslogs-stream-prefix = "pipeline"
      }
    }
  }])
}

output "ecs_cluster" { value = aws_ecs_cluster.main.name }
output "subnet_ids" { value = data.aws_subnets.default.ids }
output "sg_id" { value = aws_security_group.pipeline.id }
