
resource "aws_ssm_parameter" "sim_clock" {
  name  = "/gs-${local.environment}/rossmann/sim-cutoff"
  type  = "String"
  value = "2015-02-01"

  lifecycle {
    ignore_changes = [value]
  }
}
