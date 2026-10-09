/**
 * main.js - Electron 桌面应用外壳
 *
 * 它干的事很简单：
 *   1. 开一个桌面窗口
 *   2. 顺手把 Python 后端（web_server.py）拉起来
 *   3. 等后端就绪后，把窗口指向 http://127.0.0.1:8765
 *   4. 关窗口时把 Python 进程一起收掉（不留后台垃圾）
 *
 * 为什么要把 Python 拉起来，而不是让你手动开？
 *   因为「桌面应用」的意义就是双击一个图标就能用。
 *   如果还要你另开命令行跑 python web_server.py，那就不叫桌面应用了。
 *
 * 想调试界面的话：设环境变量 DESKTOP_NO_BACKEND=1 再启动，
 * 它会连你已经手动开着的那个服务，不会重复拉起。
 */
const { app, BrowserWindow, shell, dialog } = require("electron");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");

// 保险：如果环境里带着 ELECTRON_RUN_AS_NODE=1，Electron 会退化成「纯 Node 模式」，
// 不加载图形界面，于是 app 变成 undefined，报
//   TypeError: Cannot read properties of undefined (reading 'requestSingleInstanceLock')
// 这个变量一般是别的工具留下的。这里直接删掉，避免排查半天。
if (process.env.ELECTRON_RUN_AS_NODE) {
  console.log("[desktop] 检测到 ELECTRON_RUN_AS_NODE，已清除（否则窗口起不来）");
  delete process.env.ELECTRON_RUN_AS_NODE;
}

const HOST = "127.0.0.1";
const PORT = Number(process.env.DESKTOP_PORT || 8765);
const URL = `http://${HOST}:${PORT}`;
const NO_BACKEND = process.env.DESKTOP_NO_BACKEND === "1";

let pyProc = null;
let win = null;

/** 找到项目根目录和 Python 解释器 */
function paths() {
  // desktop/ 的上一级就是项目根目录
  const root = path.resolve(__dirname, "..");
  const venvPy = path.join(root, "venv", "Scripts", "python.exe");
  const py = fs.existsSync(venvPy) ? venvPy : "python";
  return { root, py };
}

/** 拉起 Python 后端 */
function startBackend() {
  if (NO_BACKEND) {
    console.log("[desktop] DESKTOP_NO_BACKEND=1，跳过拉起后端（假定你已经手动开好）");
    return;
  }
  const { root, py } = paths();
  const script = path.join(root, "web_server.py");
  if (!fs.existsSync(script)) {
    dialog.showErrorBox("找不到后端脚本", `预期位置：\n${script}`);
    return;
  }

  console.log(`[desktop] 启动后端：${py} ${script}`);
  const args = [script, "--port", String(PORT)];
  // 默认开摄像头。设 DESKTOP_NO_CAMERA=1 可以临时关掉（没摄像头时调试用）。
  // 注意：早期版本这里写死了 --no-camera，导致界面里永远是一片黑 —— 那是个 bug，
  // 摄像头明明可用却被自己禁掉了。
  if (process.env.DESKTOP_NO_CAMERA === "1") {
    args.push("--no-camera");
    console.log("[desktop] DESKTOP_NO_CAMERA=1，本次不开摄像头");
  }
  pyProc = spawn(py, args, {
    cwd: root,
    windowsHide: true,           // 不弹黑框
    env: { ...process.env, PYTHONIOENCODING: "utf-8" },
  });

  pyProc.stdout.on("data", (d) => process.stdout.write("[py] " + d));
  pyProc.stderr.on("data", (d) => process.stderr.write("[py!] " + d));
  pyProc.on("exit", (code) => {
    console.log(`[desktop] 后端退出，code=${code}`);
    pyProc = null;
  });
}

/** 反复探测后端，直到它响应或超时 */
async function waitForBackend(timeoutMs = 60000) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    try {
      const r = await fetch(`${URL}/stats`, { cache: "no-store" });
      if (r.ok) {
        console.log("[desktop] 后端已就绪");
        return true;
      }
    } catch (e) {
      /* 还没起来，继续等 */
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  return false;
}

async function createWindow() {
  win = new BrowserWindow({
    width: 1280,
    height: 820,
    minWidth: 700,
    minHeight: 560,
    title: "华小牛 · 视觉识别工作台",
    backgroundColor: "#14161a",
    autoHideMenuBar: true,
    webPreferences: {
      nodeIntegration: false,     // 安全：页面里不能直接调 Node
      contextIsolation: true,
    },
  });

  // 页面里的外链用系统浏览器打开，不要在应用窗口里跳走
  win.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });

  const ready = await waitForBackend();
  if (ready) {
    win.loadURL(URL);
  } else {
    // 后端起不来时，别只给一张白屏 —— 把原因和排查方向直接写在窗口里
    win.loadURL(
      "data:text/html;charset=utf-8," +
        encodeURIComponent(`
        <html><body style="font-family:Microsoft YaHei;background:#14161a;color:#e6e8eb;padding:40px;line-height:1.9">
        <h2>⚠️ 后端服务没起来</h2>
        <p>桌面应用需要本机的 Python 服务配合，等了一会儿没等到它。</p>
        <p><b>可以这样排查：</b></p>
        <ol>
          <li>在项目目录手动运行：<code>venv\\Scripts\\python.exe web_server.py --no-camera</code></li>
          <li>看它报什么错（最常见是 Ollama 没启动）</li>
          <li>如果只是想看界面：先手动把上面那条跑起来，再设 <code>DESKTOP_NO_BACKEND=1</code> 重启本应用</li>
        </ol>
        <p style="color:#8b93a1">地址：${URL}</p>
        </body></html>`)
    );
  }

  win.on("closed", () => {
    win = null;
  });
}

// 只允许开一个实例：第二次双击时激活已有窗口，而不是再开一个
const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (win) {
      if (win.isMinimized()) win.restore();
      win.focus();
    }
  });

  app.whenReady().then(() => {
    startBackend();
    createWindow();
    app.on("activate", () => {
      if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
  });

  app.on("window-all-closed", () => {
    app.quit();
  });

  // 退出前把 Python 收掉，否则会留一个后台进程占着端口
  app.on("before-quit", () => {
    if (pyProc) {
      console.log("[desktop] 关闭后端进程");
      try {
        pyProc.kill();
      } catch (e) {
        /* 忽略 */
      }
      pyProc = null;
    }
  });
}
