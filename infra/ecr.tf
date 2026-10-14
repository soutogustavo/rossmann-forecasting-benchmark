
resource "aws_ecr_repository" "pipeline" {
  name                 = "gs-${local.environment}-rossmann-pipeline"
  image_tag_mutability = "IMMUTABLE"
  force_delete         = local.environment == "dev"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "pipeline" {
  repository = aws_ecr_repository.pipeline.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep only the 3 most recent images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 3
      }
      action = { type = "expire" }
    }]
  })
}

output "ecr_repository_url" {
  value = aws_ecr_repository.pipeline.repository_url
}
