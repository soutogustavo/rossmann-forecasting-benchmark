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

resource "aws_s3_bucket" "sales" {
  bucket = "gs-${local.environment}-rossmann-sales-forecasting"
  tags = {
    purpose = "ML Pipeline Artifacts for Rossmann Sales Forecasting"
  }
}

resource "aws_s3_object" "prefixes" {
  for_each = toset(["models/", "data-raw/", "data-processed/", "predictions/"])

  bucket  = aws_s3_bucket.sales.id
  key     = each.key
  content = ""

}
