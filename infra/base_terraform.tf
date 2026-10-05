terraform {
  required_version = ">= 1.10"

  cloud {
    organization = "gsouto-labs"
    workspaces {
      tags = ["rossmann-sales-forecasting"]
    }
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
    tfe = {
      source  = "hashicorp/tfe"
      version = "~> 0.60"
    }
  }
}

locals {
  environment = trimprefix(terraform.workspace, "rossmann-sales-forecasting-")
}

provider "aws" {
  region = "eu-central-1"

  default_tags {
    tags = {
      project    = "gs-rossmann-sales-forecasting"
      managed_by = "terraform"
    }
  }
}

data "tfe_outputs" "platform" {
  organization = "gsouto-labs"
  workspace    = "ml-platform-aws"
}

resource "aws_s3_bucket" "sales" {
  bucket = "gs-${local.environment}-rossmann-sales-forecasting"
  tags = {
    purpose = "ML Pipeline Artifacts for Rossmann Sales Forecasting"
  }
}

resource "aws_s3_object" "prefixes" {
  for_each = toset([
    "data-raw/",
    "data-store/"
  ])

  bucket  = aws_s3_bucket.sales.id
  key     = each.key
  content = ""

}

output "tracking_uri" {
  value = data.tfe_outputs.platform.nonsensitive_values.mlflow_tracking_uri
}
