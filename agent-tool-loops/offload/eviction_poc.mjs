// From agent-tool-loops/, run: node offload/eviction_poc.mjs (Node 24.19.0).
// Inspect observed.before.read_evictions and observed.after.steps[1].task_marker_visible.
// The driver follows filenames; the pinned middleware decides what the reader sees.
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import crypto from 'node:crypto';
import { stripTypeScriptTypes } from 'node:module';
import { fileURLToPath } from 'node:url';
const root = path.dirname(fileURLToPath(import.meta.url));
const sha = b => crypto.createHash('sha256').update(b).digest('hex');
const manifest = JSON.parse(fs.readFileSync(path.join(root,'source/manifest.json')));
for (const row of manifest.files) {
  if (sha(fs.readFileSync(path.join(root,'source',row.path))) !== row.sha256)
    throw new Error(`Source hash mismatch: ${row.path}`);
}
class ToolMessage {
  constructor(fields) { Object.assign(this, fields); }
  static isInstance(value) { return value instanceof ToolMessage; }
}
class Command { constructor(fields) {Object.assign(this, fields);} }
function load(label) {
  const state = {files:{}, messages:[]};
  let schema;
  schema = new Proxy(function(){return schema;},{get:()=>schema, apply:()=>schema});
  const ctx = vm.createContext({z:schema,ToolMessage,Command,isCommand:x=>x instanceof Command,
    getCurrentTaskInput:()=>state,isSandboxBackend:()=>false,createMiddleware:x=>x,
    tool:(callback,meta)=>({...meta,invoke:input=>callback(input,{})})});
  for (const name of ['utils','state','fs']) {
    const source = fs.readFileSync(path.join(root,'source',`${label}-${name}.ts`),'utf8');
    const js = stripTypeScriptTypes(source,{mode:'strip'})
      .replace(/^import[\s\S]*?;\n/gm,'').replace(/^export /gm,'');
    vm.runInContext(js,ctx);
  }
  const middleware = vm.runInContext('createFilesystemMiddleware({backend: s=>new StateBackend(s)})',ctx);
  return {state,middleware,apply:update=>{
    ctx.oldFiles=state.files;ctx.newFiles=update;
    state.files=vm.runInContext('fileDataReducer(oldFiles,newFiles)',ctx);
  }};
}
async function run(label, input, onlyExemptRead = false) {
  const env=load(label), steps=[];
  let filePath=null;
  for(let i=0;i<7;i++) {
    const name=i===0?'synthetic_report':'read_file';
    const id=`step-${i}`;
    const content=i===0?input:await env.middleware.tools.find(t=>t.name==='read_file').invoke({file_path:filePath});
    const request={state:env.state,toolCall:{name,id},config:{}};
    const handler=async()=>new ToolMessage({content,tool_call_id:id,name});
    // One-factor counterfactual on unchanged parent: only bypass eviction for read_file.
    const result=onlyExemptRead && name==='read_file'?await handler():await env.middleware.wrapToolCall(request,handler);
    const update=result instanceof Command?result.update:null;
    const newPaths=Object.keys(update?.files??{});
    if(update?.files) env.apply(update.files);
    // Only the message returned by the middleware is visible to the next model call.
    const shown=update?.messages?.[0]??result;
    steps.push({name,id,input_path:filePath,handler_chars:content.length,
      returned_chars:shown.content.length,evicted:newPaths.length>0,new_paths:newPaths,
      task_marker_visible:shown.content.includes('FINAL_STATUS=COMPLETE'),
      file_count:Object.keys(env.state.files).length});
    if(!newPaths.length)break;
    filePath=newPaths[0];
  }
  return {steps,evictions:steps.filter(s=>s.evicted).length,
    read_evictions:steps.filter(s=>s.name==='read_file'&&s.evicted).length,
    content_received:steps.at(-1).task_marker_visible,
    driver_bound_hit:steps.length===7&&steps.at(-1).evicted};
}
const large=Array.from({length:5},(_,i)=>'x'.repeat(19000)+(i===4?'FINAL_STATUS=COMPLETE':'')+'x'.repeat(1000)).join('\n');
const short='rows=42\nFINAL_STATUS=COMPLETE';
// The short-report controls verify that ordinary reads still return their contents.
const observed={before:await run('eviction-before',large),after:await run('eviction-after',large),
  parent_read_exemption_only:await run('eviction-before',large,true),
  short_before:await run('eviction-before',short),short_after:await run('eviction-after',short)};
const checks={
  initial_custom_result_still_evicted:observed.before.steps[0].evicted&&observed.after.steps[0].evicted,
  native_parent_repeatedly_reoffloads:observed.before.read_evictions===6&&observed.before.driver_bound_hit,
  native_patch_delivers_read_result:observed.after.evictions===1&&observed.after.read_evictions===0&&observed.after.content_received,
  single_change_is_sufficient:observed.parent_read_exemption_only.evictions===1&&observed.parent_read_exemption_only.content_received,
  normal_task_content_preserved:observed.short_before.evictions===0&&observed.short_after.evictions===0&&observed.short_before.content_received&&observed.short_after.content_received,
};
const report={case_id:'C01',evidence_class:'native_middleware_conditional_reoffload_cycle',
  recorded_at:new Date().toISOString(),node:process.version,
  before_revision:'73b4ada33066d84bfac95162cf4153b54c66cf7b',after_revision:'ef821a7d7bde3cd862ea5d7a95a3d86870b10409',
  fixture:{chars:large.length,lines:5,sha256:sha(large)},observed,checks,all_checks_pass:Object.values(checks).every(Boolean),
  limitations:['Read selections come from a bounded deterministic driver, not a model; this proves the conditional self-offload mechanism, not probability of a natural model loop.',
    'Tool/Command message identity, tool registration, schema construction and graph state access are local doubles; middleware callbacks, state backend, formatters and file reducer execute native pinned source.',
    'PR139 is a broad patch; parent_read_exemption_only separately isolates read_file eviction exemption as sufficient for this fixture.',
    'Original issue does not pin affected build; this ancestor has the reported mechanism, not proven identical deployed environment.',
    'No checkpoint or DeepSeek provider runs in this native component PoC; separate transfer experiments remain necessary.']};
if(process.argv[2])fs.writeFileSync(process.argv[2],JSON.stringify(report,null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify(report,null,2));
if(!report.all_checks_pass)process.exitCode=1;
