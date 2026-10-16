
variable "alert_email" {
  description = "E-mail that receives pipeline failure alerts"
  type        = string
}

resource "aws_sns_topic" "alerts" {
  name = "gs-${local.environment}-rossmann-alerts"
}

resource "aws_sns_topic_subscription" "alerts_email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

resource "aws_cloudwatch_event_rule" "task_failed" {
  name        = "gs-${local.environment}-rossmann-task-failed"
  description = "Task of pipeline Rossmann finished with exit code != 0"

  event_pattern = jsonencode({
    source        = ["aws.ecs"]
    "detail-type" = ["ECS Task State Change"]
    detail = {
      clusterArn = [aws_ecs_cluster.main.arn]
      lastStatus = ["STOPPED"]
      # Alert for 1 (failure) and 2 (gate rejected). To alert only on failure: [1]
      containers = {
        exitCode = [{ "anything-but" = 0 }]
      }
    }
  })
}

resource "aws_cloudwatch_event_target" "task_failed_email" {
  rule = aws_cloudwatch_event_rule.task_failed.name
  arn  = aws_sns_topic.alerts.arn

  input_transformer {
    input_paths = {
      exit   = "$.detail.containers[0].exitCode"
      reason = "$.detail.stoppedReason"
      task   = "$.detail.taskArn"
    }
    input_template = "\"Rossmann pipeline (${local.environment}) ended with exit code <exit> (1 = failure, 2 = gate rejected). Reason: <reason>. Task: <task>. Diagnostic: scripts/ops/check_task.sh <task>\""
  }
}

data "aws_iam_policy_document" "alerts_topic" {
  statement {
    actions   = ["sns:Publish"]
    resources = [aws_sns_topic.alerts.arn]
    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com"]
    }
    condition {
      test     = "ArnEquals"
      variable = "aws:SourceArn"
      values   = [aws_cloudwatch_event_rule.task_failed.arn]
    }
  }
}

resource "aws_sns_topic_policy" "alerts" {
  arn    = aws_sns_topic.alerts.arn
  policy = data.aws_iam_policy_document.alerts_topic.json
}
