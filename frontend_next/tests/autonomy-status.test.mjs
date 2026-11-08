import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import ts from 'typescript';
const source = await readFile(new URL('../app/autonomy-status.ts', import.meta.url), 'utf8');
const compiled = ts.transpileModule(source, {compilerOptions:{module:ts.ModuleKind.ESNext}}).outputText;
const { statusLabel, alertLabel, actionAllowed, jobKindLabel, jobStatusLabel } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`);
test('心跳过期和配置分开显示', () => {
  assert.equal(statusLabel({configured:true,running:false,heartbeat_status:'stale'}), '已配置，调度心跳已过期');
  assert.equal(statusLabel({configured:false,running:false}), '未配置后台调度');
  assert.equal(statusLabel({configured:true,running:true}), '已配置，调度运行中');
});
test('告警与不确定动作状态', () => {
  assert.equal(alertLabel('failed'), '发送失败');
  assert.equal(alertLabel('pending'), '待送达');
  assert.equal(alertLabel('acknowledged'), '已人工确认');
  assert.equal(actionAllowed('uncertain'), false);
  assert.equal(actionAllowed('pending'), true);
});

test('送达及任务状态均显示中文', () => {
  assert.equal(alertLabel('delivered'), '已送达');
  assert.equal(alertLabel('sending'), '发送中');
  assert.equal(jobKindLabel('planned_action'), '计划动作');
  assert.equal(jobKindLabel('effect_check'), '效果复查');
  assert.equal(jobStatusLabel('awaiting_schedule'), '待安排');
  assert.equal(jobStatusLabel('executing'), '执行中');
  assert.equal(jobStatusLabel('unverified'), '待核实');
});
