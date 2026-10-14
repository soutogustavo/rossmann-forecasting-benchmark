data "aws_iam_policy_document" "ecs_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "execution" {
  name               = "gs-${local.environment}-rossmann-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "task" {
  name               = "gs-${local.environment}-rossmann-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

data "aws_iam_policy_document" "task" {

  statement {
    sid       = "DataBucketList"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.sales.arn]
  }
  statement {
    sid       = "DataBucketObjects"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = ["${aws_s3_bucket.sales.arn}/*"]
  }

  statement {
    sid       = "MlflowApp"
    actions   = ["sagemaker:CallMlflowAppApi", "sagemaker-mlflow:*"]
    resources = [data.tfe_outputs.platform.nonsensitive_values.mlflow_tracking_uri]
  }
  statement {
    sid       = "MlflowArtifactsList"
    actions   = ["s3:ListBucket", "s3:GetBucketLocation"]
    resources = ["arn:aws:s3:::${data.tfe_outputs.platform.nonsensitive_values.artifact_bucket_name}"]
  }
  statement {
    sid       = "MlflowArtifactsReadWrite"
    actions   = ["s3:GetObject", "s3:PutObject"]
    resources = ["arn:aws:s3:::${data.tfe_outputs.platform.nonsensitive_values.artifact_bucket_name}/*"]
  }
}

resource "aws_iam_role_policy" "task" {
  name   = "pipeline-access"
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.task.json
}
