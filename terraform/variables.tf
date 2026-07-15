variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "us-east-1"
}

variable "namespace_name" {
  description = "Redshift Serverless namespace name"
  type        = string
  default     = "churn-pipeline"
}

variable "workgroup_name" {
  description = "Redshift Serverless workgroup name"
  type        = string
  default     = "churn-pipeline-wg"
}

variable "db_name" {
  description = "Database name inside the Redshift namespace"
  type        = string
  default     = "churn"
}

variable "admin_username" {
  description = "Redshift admin username"
  type        = string
  default     = "churn_admin"
}

variable "admin_password" {
  description = "Redshift admin password (pass via TF_VAR_admin_password env var, never commit it)"
  type        = string
  sensitive   = true
}

variable "allowed_cidr" {
  description = "Your IP in CIDR form (e.g. 1.2.3.4/32) allowed to connect to Redshift"
  type        = string
}

variable "environment" {
  description = "Environment tag"
  type        = string
  default     = "portfolio"
}
