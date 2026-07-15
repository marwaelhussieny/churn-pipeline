output "workgroup_endpoint" {
  value = aws_redshiftserverless_workgroup.churn.endpoint
}

output "namespace_name" {
  value = aws_redshiftserverless_namespace.churn.namespace_name
}

output "db_name" {
  value = var.db_name
}

output "connection_string_template" {
  description = "Copy this, fill in password, use as REDSHIFT_CONN_STRING"
  value       = "postgresql://${var.admin_username}:<PASSWORD>@<ENDPOINT_HOST>:5439/${var.db_name}"
}
