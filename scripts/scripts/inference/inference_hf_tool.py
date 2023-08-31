import argparse
import json, os
from llama_condense_monkey_patch import (
    replace_llama_with_condense,
)
# ratio = 4 means the sequence length is expanded by 4, remember to change the model_max_length to 8192 (2048 * ratio) for ratio = 4
replace_llama_with_condense(ratio=4)

from llama_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)
replace_llama_attn_with_flash_attn()
DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。"""

TEMPLATE = (
    "[INST] <<SYS>>\n"
    "{system_prompt}\n"
    "<</SYS>>\n\n"
    "{instruction} [/INST]"
)

parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default=None, type=str, required=True)
parser.add_argument('--lora_model', default=None, type=str, help="If None, perform inference on the base model")
parser.add_argument('--tokenizer_path', default=None, type=str)
parser.add_argument('--data_file', default=None, type=str,
                    help="A file that contains instructions (one instruction per line)")
parser.add_argument('--with_prompt', action='store_true', help="wrap the input with the prompt automatically")
parser.add_argument('--interactive', action='store_true', help="run in the instruction mode (single-turn)")
parser.add_argument('--predictions_file', default='./predictions.json', type=str)
parser.add_argument('--gpus', default="0", type=str)
parser.add_argument('--only_cpu', action='store_true', help='only use CPU for inference')
parser.add_argument('--alpha', type=str, default="1.0",
                    help="The scaling factor of NTK method, can be a float or 'auto'. ")
parser.add_argument('--load_in_8bit', action='store_true', help="Load the LLM in the 8bit mode")
parser.add_argument('--load_in_4bit', action='store_true', help="Load the LLM in the 4bit mode")
parser.add_argument("--use_vllm", action='store_true', help="Use vLLM as back-end LLM service.")
parser.add_argument('--system_prompt', type=str, default=DEFAULT_SYSTEM_PROMPT,
                    help="The system prompt of the prompt template.")
parser.add_argument('--negative_prompt', type=str, default=None, help="Negative prompt in CFG sampling.")
parser.add_argument('--guidance_scale', type=float, default=1.0,
                    help="The guidance scale for CFG sampling. CFG is enabled by setting `guidance_scale > 1`.")
args = parser.parse_args()

if args.guidance_scale > 1:
    try:
        from transformers.generation import UnbatchedClassifierFreeGuidanceLogitsProcessor
    except ImportError:
        raise ImportError(
            "Please install the latest transformers (commit equal or later than d533465) to enable CFG sampling.")

if args.load_in_8bit and args.load_in_4bit:
    raise ValueError("Only one quantization method can be chosen for inference. Please check your arguments")
if args.only_cpu is True:
    args.gpus = ""
    if args.load_in_8bit or args.load_in_4bit:
        raise ValueError("Quantization is unavailable on CPU.")
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import GenerationConfig
from transformers import BitsAndBytesConfig
from peft import PeftModel

if args.use_vllm:
    from vllm import LLM, SamplingParams

import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)



generation_config = GenerationConfig(
    temperature=0.2,
    top_k=40,
    top_p=0.9,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.1,
    max_new_tokens=8192
)

sample_data = ["\n    现在希望你扮演代码分析人员, 你可以独立思考分析，或调用以下工具（函数），根据任务描述指示来实现以下任务。\n    我会给你任务描述，然后可以开始执行任务。\n    在每个步骤中，你需要思考分析当前状态以及下一步要做什么，当满足工具使用条件时才调用对应工具，否则请自行思考完成分析。你的输出应该遵循以下格式：\n    -------------------\n    Thought: <思考应该执行的步骤>\n    Action: <执行分析动作或工具名称>\n    Action Input: <分析或工具调用的输入>\n    Observation: <分析结果/工具调用返回的内容>\n    -------------------\n    在调用或自行分析之后，您将获得分析结果，并处于新的状态中。\n    然后，您将分析当前状态，然后决定下一步该做什么\n    经过许多（思考-分析）配对后，最终执行任务，然后可以给出最终答案。\n    注意：\n    1) 状态变更是不可逆的，您不能回到之前的任何状态，如果您想重新开始任务，请说 \"我放弃并重新开始\"。\n    2) 所有的思考都要简洁，最多不超过 5 句话。\n    3) 您可以尝试多次，因此，如果您的计划是持续尝试某些条件，您可以每次尝试一个条件。\n    4) 在任务结束时，务必调用 `finish` 工具。最终的答案应包含足够的信息以展示给用户。如果您无法完成任务，或者发现函数调用始终失败（函数当前无效），请使用工具 finish->give_up_and_restart。\n\n    你可以使用如下的工具:\n    1.callback_pycode_exec: 当发现代码存在source和sink点、需要进行精准代码流转推导时，可以调用此工具进行分析。\n    2.finish, 这个工具用于返回任务的最终分析结果。当最终结果已经得到或者需要放弃分析时，它将被调用。\n    注意，你不擅长分析数学运算、代码精确分析，请尽量使用提供的‘callback_pycode_exec‘工具解决相关问题。\n    以下是工具调用的方式说明：\n    - {'name': 'callback_pycode_exec', 'description': 'python污点变量传播代码分析器', 'parameters': {'type': 'object', 'properties': {'param': {'type': 'string', 'description': '污点初始变量'}, 'code_str': {'type': 'string', 'description': '待分析的传播链路python代码'}}, 'required': ['param', 'code_str']}}\n\n    - {'name': 'finish', 'description': '如果您认为已经得到了可以回答任务的结果，请调用此函数提供最终答案；如果找不到污点变量或危险函数，认为无需进行分析，请调用此函数；如果您认识到在当前状态下无法继续执行任务，请调用此函数重新开始。请记住：您必须在尝试结束时始终调用此函数，而且将向用户显示的唯一部分是最终答案，因此它应包含足够的信息。', 'parameters': {'type': 'object', 'properties': {'return_type': {'type': 'string', 'enum': ['give_answer', 'give_up_and_restart']}, 'final_answer': {'type': 'string', 'description': '如果 \"return_type\"==\"give_answer\"，则这是您想要向用户提供的最终答案'}}, 'required': ['return_type']}}\n    \n    接下来，我会给你任务描述，然后你就可以开始任务了，记得在给出最终答案前多按照指定格式进行一步一步的推理。\n    \n    任务描述：\n    现在为你提供如下一段Java代码，请逐一按照以下步骤进行分析：\n    1）首先请分析代码中是否包含bar和param变量，变量查找要靠你自己而不能调工具，需要你给出分析结果：如果不存在bar或param变量，请在下一步调用`finish`工具直接告知不存在变量然后结束；如果bar和param变量都存在执行下一步；\n    2）然后请分析变量param传递到bar的过程，将包含关键步骤代码（包含条件判断、赋值、map/list操作、函数调用等复杂调用逻辑）转换成等价可运行python语言代码（注意不能使用函数或者类定义，所有代码放在最外层同一作用域，不符合python语法的代码请做等价转换，补充必要的import导包，对于依赖的变量要么保留原代码中取值要么随机初始化赋值），转换完代码记得执行后续分析步骤；\n    3）完成上一步python代码转换后，调用‘callback_pycode_exec’工具进行精准传播分析；\n    4) 如果在工具分析结果多次出错之后再尝试不依赖工具自行分析变量路径是否可达，在此之前请首先使用工具‘callback_pycode_exec’；\n    5）如果分析结果提示变量可以传递，回答“param可以传递到bar”；否则回答“param不可以传递到bar”。\n    PS：在分析执行结束之前不用做原因解释，如果认为已经得到了最终结果，一定请调用‘finish’工具输出分析结果然后再结束。\n\npackage org.glowroot.agent.plugin.quartz;\n\nimport java.util.Date;\n\nimport org.junit.After;\nimport org.junit.AfterClass;\nimport org.junit.BeforeClass;\nimport org.junit.Test;\nimport org.quartz.JobDetail;\nimport org.quartz.Scheduler;\nimport org.quartz.Trigger;\nimport org.quartz.impl.StdSchedulerFactory;\n\nimport org.glowroot.agent.it.harness.AppUnderTest;\nimport org.glowroot.agent.it.harness.Container;\nimport org.glowroot.agent.it.harness.Containers;\nimport org.glowroot.wire.api.model.TraceOuterClass.Trace;\n\nimport static java.util.concurrent.TimeUnit.SECONDS;\nimport static org.assertj.core.api.Assertions.assertThat;\n\npublic class QuartzPluginIT {\n\n    private static Container container;\n\n    @BeforeClass\n    public static void setUp() throws Exception {\n        container = Containers.create();\n    }\n\n    @AfterClass\n    public static void tearDown() throws Exception {\n        container.close();\n    }\n\n    @After\n    public void afterEachTest() throws Exception {\n        container.checkAndReset();\n    }\n\n    @Test\n    public void shouldCaptureJobExecution() throws Exception {\n        Trace trace = container.execute(ExecuteJob.class);\n        Trace.Header header = trace.getHeader();\n        assertThat(header.getTransactionType()).isEqualTo(\"Background\");\n        assertThat(header.getTransactionName()).isEqualTo(\"Quartz job: ajob\");\n        assertThat(header.getHeadline()).isEqualTo(\"Quartz job: ajob\");\n    }\n\n    private static JobDetail createJob1x() throws Exception {\n        Class<?> clazz = Class.forName(\"org.quartz.JobDetail\");\n        Object param = clazz.newInstance();\n        clazz.getMethod(\"setName\", String.class).invoke(param, \"ajob\");\n        clazz.getMethod(\"setJobClass\", Class.class).invoke(param, TestJob.class);\n        return (JobDetail) param;\n    }\n\n    private static JobDetail createJob2x() throws Exception {\n        Class<?> clazz = Class.forName(\"org.quartz.JobBuilder\");\n        Object jobBuilder = clazz.getMethod(\"newJob\", Class.class).invoke(null, TestJob.class);\n        clazz.getMethod(\"withIdentity\", String.class, String.class).invoke(jobBuilder, \"ajob\",\n                \"agroup\");\n        return (JobDetail) clazz.getMethod(\"build\").invoke(jobBuilder);\n    }\n\n    private static Trigger createTrigger1x() throws Exception {\n        Class<?> clazz = Class.forName(\"org.quartz.SimpleTrigger\");\n        Object param = clazz.newInstance();\n        clazz.getMethod(\"setName\", String.class).invoke(param, \"atrigger\");\n        clazz.getMethod(\"setStartTime\", Date.class).invoke(param, new Date());\n        return (Trigger) param;\n    }\n\n    private static Trigger createTrigger2x() throws Exception {\n        Class<?> clazz = Class.forName(\"org.quartz.TriggerBuilder\");\n        Object triggerBuilder = clazz.getMethod(\"newTrigger\").invoke(null);\n        clazz.getMethod(\"withIdentity\", String.class, String.class).invoke(triggerBuilder,\n                \"atrigger\", \"agroup\");\n        clazz.getMethod(\"startNow\").invoke(triggerBuilder);\n        return (Trigger) clazz.getMethod(\"build\").invoke(triggerBuilder);\n    }\n\n    public static class ExecuteJob implements AppUnderTest {\n        @Override\n        public void executeApp() throws Exception {\n            Scheduler scheduler = StdSchedulerFactory.getDefaultScheduler();\n            scheduler.start();\n            JobDetail job;\n            Trigger trigger;\n            try {\n                job = createJob2x();\n                trigger = createTrigger2x();\n            } catch (ClassNotFoundException e) {\n                job = createJob1x();\n                trigger = createTrigger1x();\n            }\n            scheduler.scheduleJob(job, trigger);\n            SECONDS.sleep(1);\n            bar = scheduler;\n            bar.shutdown();\n        }\n    }\n}"]


def generate_prompt(instruction, system_prompt=DEFAULT_SYSTEM_PROMPT):
    return TEMPLATE.format_map({'instruction': instruction, 'system_prompt': system_prompt})


if __name__ == '__main__':
    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')
    if args.tokenizer_path is None:
        args.tokenizer_path = args.lora_model
        if args.lora_model is None:
            args.tokenizer_path = args.base_model


    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path, legacy=True)


    base_model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
        device_map='auto',
        cache_dir=None,
    )
    base_model.config.use_cache = False

    model_vocab_size = base_model.get_input_embeddings().weight.size(0)
    tokenizer_vocab_size = len(tokenizer)
    print(f"Vocab of the base model: {model_vocab_size}")
    print(f"Vocab of the tokenizer: {tokenizer_vocab_size}")
    if model_vocab_size != tokenizer_vocab_size:
        print("Resize model embeddings to fit tokenizer")
        base_model.resize_token_embeddings(tokenizer_vocab_size)
    if args.lora_model is not None:
        print("loading peft model")
        model = PeftModel.from_pretrained(base_model, args.lora_model, torch_dtype=load_type,
                                          device_map='auto', ).half()
    else:
        model = base_model

        if device == torch.device('cpu'):
            model.float()
        model.eval()

    # test data
    if args.data_file is None:
        examples = sample_data
    else:
        with open(args.data_file, 'r') as f:
            examples = [l.strip() for l in f.readlines()]
        print("first 10 examples:")
        for example in examples[:10]:
            print(example)

    with torch.no_grad():
        print("Start inference.")
        results = []
        for index, example in enumerate(examples):
            if args.with_prompt:
                input_text = generate_prompt(instruction=example, system_prompt=args.system_prompt)
                negative_text = None if args.negative_prompt is None else \
                    generate_prompt(instruction=example, system_prompt=args.negative_prompt)
            else:
                input_text = example
                negative_text = args.negative_prompt
            inputs = tokenizer(input_text, return_tensors="pt")  # add_special_tokens=False ?
            if args.guidance_scale == 1:
                generation_output = model.generate(
                    input_ids=inputs["input_ids"].to(device),
                    attention_mask=inputs['attention_mask'].to(device),
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    generation_config=generation_config
                )
            s = generation_output[0]
            output = tokenizer.decode(s, skip_special_tokens=True)
            print('ooooooooooooooooooooo', output)
            if args.with_prompt:
                response = output.split("[/INST]")[1].strip()
            else:
                response = output
            print(f"======={index}=======")
            print(f"Input: {example}\n")
            print(f"Output: {response}\n")

            results.append({"Input": input_text, "Output": response})

        dirname = os.path.dirname(args.predictions_file)
        os.makedirs(dirname, exist_ok=True)
        with open(args.predictions_file, 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        if args.use_vllm:
            with open(dirname + '/generation_config.json', 'w') as f:
                json.dump(generation_config, f, ensure_ascii=False, indent=2)
        else:
            generation_config.save_pretrained('./')
