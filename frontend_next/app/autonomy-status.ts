export function statusLabel(status: Record<string, any>): string {
  if (!status.configured) return '未配置后台调度';
  if (status.running) return '已配置，调度运行中';
  return status.heartbeat_status === 'stale' ? '已配置，调度心跳已过期' : '已配置，调度未运行';
}
export function alertLabel(status: string): string {
  return ({pending:'待送达', sending:'发送中', failed:'发送失败', sent:'已送达', delivered:'已送达', acknowledged:'已人工确认'} as Record<string, string>)[status] || '未知状态';
}
export function actionAllowed(status: string): boolean { return status !== 'uncertain'; }

export function jobKindLabel(kind: string): string {
  return ({planned_action:'计划动作', effect_check:'效果复查'} as Record<string, string>)[kind] || '未知任务';
}
export function jobStatusLabel(status: string): string {
  return ({pending:'待执行', awaiting_schedule:'待安排', executing:'执行中', completed:'已完成', failed:'失败', unverified:'待核实', uncertain:'结果不确定', cancelled:'已取消'} as Record<string, string>)[status] || '未知状态';
}
