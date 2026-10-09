/* 视觉识别工作台 —— 前端逻辑
   只做三件事：每 1 秒取一次状态、发消息给后端、把结果显示出来。
   没有框架、没有构建步骤 —— 这样 Electron 直接加载就行。 */

const $ = (id) => document.getElementById(id);
const chatBox = $("chat");
const input = $("input");
const sendBtn = $("send");
let busy = false;

/* ---------- 聊天显示 ---------- */
function addMsg(who, text, cls) {
  const d = document.createElement("div");
  d.className = "msg " + cls;
  const w = document.createElement("div");
  w.className = "who";
  w.textContent = who;
  const b = document.createElement("div");
  b.className = "body";
  b.textContent = text;
  d.appendChild(w);
  d.appendChild(b);
  chatBox.appendChild(d);
  chatBox.scrollTop = chatBox.scrollHeight;
}

/* ---------- 状态轮询 ---------- */
async function pollStats() {
  try {
    const r = await fetch("/stats", { cache: "no-store" });
    const s = await r.json();

    $("s-faces").textContent = s.faces ?? "-";
    $("s-known").textContent = s.known ?? "-";
    $("s-pending").textContent = s.pending ?? "-";
    $("s-fps").textContent = s.fps ?? "-";

    const prov = (s.provider ? s.provider + " / " : "") + (s.model || "");
    $("provider").textContent = prov || "";

    // 摄像头打不开时，把黑屏盖上一层说明 —— 否则用户只看到一片黑，不知道发生了什么
    const msg = $("cammsg");
    if (s.camera_ok === false && s.message) {
      msg.style.display = "flex";
      msg.textContent = s.message;
    } else {
      msg.style.display = "none";
    }

    // 物体列表
    const objs = s.objects || [];
    $("objects").textContent = objs.length
      ? "检测到：" + objs.map(o => `${o.name} ${o.conf}`).join("、")
      : "";

    // 新面孔提醒
    (s.new_faces || []).forEach(f => {
      addMsg("系统", `🆕 发现新面孔：${f.id}（相似度 ${f.score}）` +
             (f.snapshot ? `，已抓拍 ${f.snapshot}` : ""), "alert");
    });
  } catch (e) {
    $("provider").textContent = "后端未连接";
  }
}

/* ---------- 发消息 ---------- */
async function ask(text) {
  if (busy || !text.trim()) return;
  busy = true;
  sendBtn.disabled = true;
  document.querySelectorAll("button").forEach(b => b.disabled = true);

  addMsg("你", text, "user");
  addMsg("系统", "正在思考…（画面继续刷新）", "sys");

  try {
    const r = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    const d = await r.json();
    // 去掉"正在思考…"那条
    const last = chatBox.lastElementChild;
    if (last && last.textContent.includes("正在思考")) last.remove();

    if (d.reply) {
      addMsg("华小牛", d.reply, "ai");
    } else {
      addMsg("系统", "出错了：" + (d.error || "未知错误"), "alert");
    }
  } catch (e) {
    addMsg("系统", "请求失败：" + e.message, "alert");
  } finally {
    busy = false;
    sendBtn.disabled = false;
    document.querySelectorAll("button").forEach(b => b.disabled = false);
    input.focus();
  }
}

/* ---------- 按钮 ---------- */
sendBtn.addEventListener("click", () => {
  const t = input.value;
  input.value = "";
  ask(t);
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    const t = input.value;
    input.value = "";
    ask(t);
  }
});

document.querySelectorAll("[data-say]").forEach(btn => {
  btn.addEventListener("click", () => ask(btn.dataset.say));
});

document.querySelectorAll("[data-act]").forEach(btn => {
  btn.addEventListener("click", async () => {
    const act = btn.dataset.act;
    if (act === "clear") {
      chatBox.innerHTML = "";
      return;
    }
    if (act === "who") {
      const s = await (await fetch("/stats", { cache: "no-store" })).json();
      addMsg("华小牛", `当前画面里有 ${s.faces} 张脸，脸库共 ${s.known} 人，${s.pending} 张在投票中。`, "ai");
      return;
    }
    if (act === "snapshot") {
      const r = await (await fetch("/snapshot", { method: "POST" })).json();
      addMsg("系统", r.saved ? "📁 已存图：" + r.saved : "存图失败：" + (r.error || ""),
             r.saved ? "sys" : "alert");
    }
  });
});

/* ---------- 启动 ---------- */
addMsg("系统", "界面已就绪。左边是摄像头画面，右边打字就能问 AI。", "sys");
pollStats();
setInterval(pollStats, 1000);
input.focus();
