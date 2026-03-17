"""
Helper for running LibreOffice (soffice) cross-platform.

- Windows : Program Files 아래에서 soffice.exe 를 자동 탐색합니다.
- Linux   : AF_UNIX 소켓이 차단된 샌드박스 환경에서 LD_PRELOAD shim 을 적용합니다.
- macOS   : PATH 에서 soffice 를 찾습니다.

Usage:
    from soffice import run_soffice, get_soffice_env

    result = run_soffice(["--headless", "--convert-to", "pdf", "input.docx"])
"""

import os
import sys
import socket
import subprocess
import tempfile
from pathlib import Path


# ─── Windows LibreOffice 실행 파일 탐색 ──────────────────────────────────────
_WINDOWS_LO_CANDIDATES = [
    r"C:\Program Files\LibreOffice\program\soffice.exe",
    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
]

def find_soffice_exe() -> str:
    """플랫폼에 맞는 LibreOffice 실행 파일 경로를 반환합니다."""
    if sys.platform == "win32":
        for path in _WINDOWS_LO_CANDIDATES:
            if Path(path).exists():
                return path
        raise FileNotFoundError(
            "LibreOffice를 찾을 수 없습니다.\n"
            "https://www.libreoffice.org/download 에서 설치 후 다시 시도하세요.\n"
            "탐색한 경로:\n" + "\n".join(f"  {p}" for p in _WINDOWS_LO_CANDIDATES)
        )
    return "soffice"  # Linux / macOS 는 PATH 에서 탐색


# ─── 환경 변수 설정 ───────────────────────────────────────────────────────────
def get_soffice_env() -> dict:
    env = os.environ.copy()

    if sys.platform != "win32":
        # Linux headless 렌더링 플러그인 지정
        env["SAL_USE_VCLPLUGIN"] = "svp"
        if _needs_shim():
            shim = _ensure_shim()
            env["LD_PRELOAD"] = str(shim)

    return env


# ─── 실행 진입점 ──────────────────────────────────────────────────────────────
def run_soffice(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    exe = find_soffice_exe()
    env = get_soffice_env()
    return subprocess.run([exe] + args, env=env, **kwargs)


# ─── Linux shim (AF_UNIX 소켓 차단 우회) ─────────────────────────────────────
_SHIM_SO = Path(tempfile.gettempdir()) / "lo_socket_shim.so"


def _needs_shim() -> bool:
    # Windows 에는 AF_UNIX 가 없으므로 shim 불필요
    if not hasattr(socket, "AF_UNIX"):
        return False
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.close()
        return False
    except OSError:
        return True


def _ensure_shim() -> Path:
    if _SHIM_SO.exists():
        return _SHIM_SO

    src = Path(tempfile.gettempdir()) / "lo_socket_shim.c"
    src.write_text(_SHIM_SOURCE)
    subprocess.run(
        ["gcc", "-shared", "-fPIC", "-o", str(_SHIM_SO), str(src), "-ldl"],
        check=True,
        capture_output=True,
    )
    src.unlink()
    return _SHIM_SO


_SHIM_SOURCE = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <unistd.h>

static int (*real_socket)(int, int, int);
static int (*real_socketpair)(int, int, int, int[2]);
static int (*real_listen)(int, int);
static int (*real_accept)(int, struct sockaddr *, socklen_t *);
static int (*real_close)(int);
static int (*real_read)(int, void *, size_t);

static int is_shimmed[1024];
static int peer_of[1024];
static int wake_r[1024];
static int wake_w[1024];
static int listener_fd = -1;

__attribute__((constructor))
static void init(void) {
    real_socket     = dlsym(RTLD_NEXT, "socket");
    real_socketpair = dlsym(RTLD_NEXT, "socketpair");
    real_listen     = dlsym(RTLD_NEXT, "listen");
    real_accept     = dlsym(RTLD_NEXT, "accept");
    real_close      = dlsym(RTLD_NEXT, "close");
    real_read       = dlsym(RTLD_NEXT, "read");
    for (int i = 0; i < 1024; i++) {
        peer_of[i] = -1;
        wake_r[i]  = -1;
        wake_w[i]  = -1;
    }
}

int socket(int domain, int type, int protocol) {
    if (domain == AF_UNIX) {
        int fd = real_socket(domain, type, protocol);
        if (fd >= 0) return fd;
        int sv[2];
        if (real_socketpair(domain, type, protocol, sv) == 0) {
            if (sv[0] >= 0 && sv[0] < 1024) {
                is_shimmed[sv[0]] = 1;
                peer_of[sv[0]]    = sv[1];
                int wp[2];
                if (pipe(wp) == 0) { wake_r[sv[0]] = wp[0]; wake_w[sv[0]] = wp[1]; }
            }
            return sv[0];
        }
        errno = EPERM;
        return -1;
    }
    return real_socket(domain, type, protocol);
}

int listen(int sockfd, int backlog) {
    if (sockfd >= 0 && sockfd < 1024 && is_shimmed[sockfd]) { listener_fd = sockfd; return 0; }
    return real_listen(sockfd, backlog);
}

int accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen) {
    if (sockfd >= 0 && sockfd < 1024 && is_shimmed[sockfd]) {
        if (wake_r[sockfd] >= 0) { char buf; real_read(wake_r[sockfd], &buf, 1); }
        errno = ECONNABORTED;
        return -1;
    }
    return real_accept(sockfd, addr, addrlen);
}

int close(int fd) {
    if (fd >= 0 && fd < 1024 && is_shimmed[fd]) {
        int was_listener = (fd == listener_fd);
        is_shimmed[fd] = 0;
        if (wake_w[fd] >= 0) { char c = 0; write(wake_w[fd], &c, 1); real_close(wake_w[fd]); wake_w[fd] = -1; }
        if (wake_r[fd] >= 0) { real_close(wake_r[fd]); wake_r[fd]  = -1; }
        if (peer_of[fd] >= 0) { real_close(peer_of[fd]); peer_of[fd] = -1; }
        if (was_listener) _exit(0);
    }
    return real_close(fd);
}
"""


if __name__ == "__main__":
    result = run_soffice(sys.argv[1:])
    sys.exit(result.returncode)
