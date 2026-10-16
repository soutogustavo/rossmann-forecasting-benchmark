
variable "github_repo" {
  description = "GitHub repository (owner/name) authorized to assume the role"
  type        = string
  default     = "soutogustavo/rossmann-forecasting-benchmark"
}

data "aws_iam_openid_connect_provider" "github" {
  url = "https://token.actions.githubusercontent.com"
}

data "aws_iam_policy_document" "github_assume" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [data.aws_iam_openid_connect_provider.github.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repo}:environment:${local.environment}"]
    }
  }
}

resource "aws_iam_role" "github_ci" {
  name                 = "gs-${local.environment}-rossmann-github-ci"
  assume_role_policy   = data.aws_iam_policy_document.github_assume.json
  max_session_duration = 3600
}

data "aws_iam_policy_document" "github_ci" {
  statement {
    sid       = "EcrLogin"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid = "EcrPushThisRepo"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:CompleteLayerUpload",
      "ecr:DescribeImages",
      "ecr:InitiateLayerUpload",
      "ecr:PutImage",
      "ecr:UploadLayerPart",
    ]
    resources = [aws_ecr_repository.pipeline.arn]
  }

  statement {
    sid       = "VerifyDeploy"
    actions   = ["ecs:DescribeTaskDefinition"]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "github_ci" {
  name   = "ci-push"
  role   = aws_iam_role.github_ci.id
  policy = data.aws_iam_policy_document.github_ci.json
}

resource "aws_iam_role_policy_attachment" "github_ci_power_user" {
  role       = aws_iam_role.github_ci.name
  policy_arn = "arn:aws:iam::aws:policy/PowerUserAccess"
}

data "aws_iam_policy_document" "github_ci_iam" {
  statement {
    sid     = "ManageProjectRoles"
    actions = ["iam:*"]
    resources = [
      "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/gs-${local.environment}-rossmann-*",
    ]
  }

  statement {
    sid = "ReadOidcProvider"
    actions = [
      "iam:GetOpenIDConnectProvider",
      "iam:ListOpenIDConnectProviders",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "github_ci_iam" {
  name   = "terraform-iam"
  role   = aws_iam_role.github_ci.id
  policy = data.aws_iam_policy_document.github_ci_iam.json
}

output "github_ci_role_arn" {
  value = aws_iam_role.github_ci.arn
}
