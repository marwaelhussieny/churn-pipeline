terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# --- Reuse the account's default VPC, same pattern as the real estate project's RDS setup
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

resource "aws_security_group" "redshift" {
  name        = "churn-pipeline-redshift-sg"
  description = "Allow Redshift access from a single trusted IP"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "Redshift from trusted IP"
    from_port   = 5439
    to_port     = 5439
    protocol    = "tcp"
    cidr_blocks = [var.allowed_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Project     = "churn-pipeline"
    Environment = var.environment
  }
}

# --- Redshift Serverless: namespace holds the data, workgroup holds the compute.
#     Serverless (vs. a provisioned cluster) means you pay only for actual
#     query time - much more portfolio-friendly than a permanently-running cluster.
resource "aws_redshiftserverless_namespace" "churn" {
  namespace_name      = var.namespace_name
  admin_username       = var.admin_username
  admin_user_password  = var.admin_password
  db_name              = var.db_name

  tags = {
    Project     = "churn-pipeline"
    Environment = var.environment
  }
}

resource "aws_redshiftserverless_workgroup" "churn" {
  namespace_name = aws_redshiftserverless_namespace.churn.namespace_name
  workgroup_name = var.workgroup_name

  base_capacity      = 8 # minimum RPU - keeps this cheap for a portfolio project
  publicly_accessible = true

  subnet_ids         = data.aws_subnets.default.ids
  security_group_ids = [aws_security_group.redshift.id]

  tags = {
    Project     = "churn-pipeline"
    Environment = var.environment
  }
}
