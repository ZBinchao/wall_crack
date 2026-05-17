import os
import time
import threading
import queue
from datetime import datetime
from pathlib import Path

import cv2
from fastapi import FastAPI, Request, Depends, Form
from fastapi.responses import (
    StreamingResponse,
    HTMLResponse,
    RedirectResponse,
    JSONResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy.orm import Session
from ultralytics import YOLO

from database import get_db, init_db
from models import User, WorkOrder, Anomaly
from auth import login_user, logout_user, get_current_user, require_login, init_admin_user

# ============================================================
# 1. FastAPI 应用初始化
# ============================================================
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="wall-crack-secret-key-2024")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
templates.env.globals["now"] = datetime.now

# ============================================================
# 2. 加载 YOLO 模型
# ============================================================
model = YOLO(r"D:\wall_crack_m\runs\segment\train-4\weights\best.pt")

# ============================================================
# 3. 摄像头采集（保持不变）
# ============================================================
frame_queue = queue.Queue(maxsize=2)

def grab_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    while True:
        success, frame = cap.read()
        if not success:
            continue
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(frame)

threading.Thread(target=grab_camera, daemon=True).start()

# ============================================================
# 4. 视频流生成器（保持不变）
# ============================================================
def gen_original_stream():
    while True:
        frame = frame_queue.get()
        ret, jpeg = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        )

def gen_yolo_stream():
    while True:
        frame = frame_queue.get()
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()
        ret, jpeg = cv2.imencode(".jpg", annotated_frame)
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        )

# ============================================================
# 5. 数据库初始化
# ============================================================
init_db()
db = next(get_db())
init_admin_user(db)
db.close()

# ============================================================
# 6. 认证路由
# ============================================================
@app.get("/login")
def login_page(request: Request):
    if get_current_user(request):
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@app.post("/login")
def login_submit(request: Request, username: str = Form(), password: str = Form()):
    db = next(get_db())
    if login_user(request, username, password, db):
        db.close()
        return RedirectResponse(url="/", status_code=302)
    db.close()
    return templates.TemplateResponse(
        "login.html", {"request": request, "error": "用户名或密码错误"}
    )


@app.get("/live")
def live_page(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(
        "live.html", {"request": request, "active": "live", "user": user}
    )


@app.get("/logout")
def logout(request: Request):
    logout_user(request)
    return RedirectResponse(url="/login", status_code=302)

# ============================================================
# 7. 已有视频流路由（保持不变）
# ============================================================
@app.get("/video/original")
def video_original():
    return StreamingResponse(
        gen_original_stream(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/video/yolo")
def video_yolo():
    return StreamingResponse(
        gen_yolo_stream(), media_type="multipart/x-mixed-replace; boundary=frame"
    )

# ============================================================
# 8. 仪表盘
# ============================================================
@app.get("/")
def dashboard(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(
        "dashboard.html", {"request": request, "active": "dashboard", "user": user}
    )


@app.get("/dashboard-data")
def dashboard_data(request: Request):
    user = get_current_user(request)
    if not user:
        return JSONResponse({"error": "未登录"}, status_code=401)
    db = next(get_db())
    total_orders = db.query(WorkOrder).count()
    completed_orders = db.query(WorkOrder).filter(WorkOrder.status == "completed").count()
    total_anomalies = db.query(Anomaly).count()
    total_images = db.query(Anomaly).count()

    # 严重程度分布
    severity_stats = {"low": 0, "medium": 0, "high": 0}
    for s, c in db.query(Anomaly.severity, Anomaly.id).all():
        pass
    from sqlalchemy import func as sqlfunc
    severity_counts = (
        db.query(Anomaly.severity, sqlfunc.count(Anomaly.id))
        .group_by(Anomaly.severity)
        .all()
    )
    for sev, cnt in severity_counts:
        severity_stats[sev] = cnt

    # 每日趋势（最近7天）
    daily_counts = {}
    from datetime import timedelta
    today = datetime.now().date()
    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        daily_counts[day.strftime("%m-%d")] = 0
    anomalies_by_day = (
        db.query(sqlfunc.date(Anomaly.created_at), sqlfunc.count(Anomaly.id))
        .filter(Anomaly.created_at >= today - timedelta(days=6))
        .group_by(sqlfunc.date(Anomaly.created_at))
        .all()
    )
    for day_str, cnt in anomalies_by_day:
        key = datetime.strptime(day_str, "%Y-%m-%d").strftime("%m-%d") if isinstance(day_str, str) else day_str.strftime("%m-%d")
        if key in daily_counts:
            daily_counts[key] = cnt

    db.close()
    return {
        "total_orders": total_orders,
        "completed_orders": completed_orders,
        "total_anomalies": total_anomalies,
        "total_images": total_images,
        "severity_stats": severity_stats,
        "daily_trend": daily_counts,
    }

# ============================================================
# 9. 工单管理
# ============================================================
def require_login_dep(request: Request):
    user = get_current_user(request)
    if not user:
        return None
    return user


@app.get("/work-orders")
def work_orders_page(
    request: Request, page: int = 1, search: str = "", db: Session = Depends(get_db)
):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    per_page = 10
    query = db.query(WorkOrder)
    if search:
        query = query.filter(
            WorkOrder.title.contains(search) | WorkOrder.location.contains(search)
        )
    query = query.order_by(WorkOrder.created_at.desc())
    total = query.count()
    orders = query.offset((page - 1) * per_page).limit(per_page).all()
    users = db.query(User).all()
    return templates.TemplateResponse(
        "work_orders.html",
        {
            "request": request,
            "active": "work_orders",
            "orders": orders,
            "users": users,
            "page": page,
            "total": total,
            "per_page": per_page,
            "search": search,
            "user": user,
        },
    )


@app.get("/work-orders/new")
def new_work_order_page(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    users = db.query(User).all()
    return templates.TemplateResponse(
        "work_order_form.html",
        {"request": request, "active": "work_orders", "order": None, "users": users, "user": user},
    )


@app.post("/work-orders/new")
def create_work_order(
    request: Request,
    title: str = Form(),
    location: str = Form(""),
    inspector_id: int = Form(0),
    description: str = Form(""),
    db: Session = Depends(get_db),
):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    order = WorkOrder(
        title=title,
        location=location,
        inspector_id=inspector_id if inspector_id else None,
        description=description,
    )
    db.add(order)
    db.commit()
    return RedirectResponse(url="/work-orders", status_code=302)


@app.get("/work-orders/{order_id}")
def work_order_detail(
    request: Request, order_id: int, db: Session = Depends(get_db)
):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    order = db.query(WorkOrder).filter(WorkOrder.id == order_id).first()
    if not order:
        return HTMLResponse("工单不存在", status_code=404)
    anomalies = (
        db.query(Anomaly)
        .filter(Anomaly.work_order_id == order_id)
        .order_by(Anomaly.created_at.desc())
        .all()
    )
    inspector = db.query(User).filter(User.id == order.inspector_id).first()
    return templates.TemplateResponse(
        "work_order_detail.html",
        {
            "request": request,
            "active": "work_orders",
            "order": order,
            "anomalies": anomalies,
            "inspector": inspector,
            "user": user,
        },
    )


@app.get("/work-orders/{order_id}/edit")
def edit_work_order_page(
    request: Request, order_id: int, db: Session = Depends(get_db)
):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    order = db.query(WorkOrder).filter(WorkOrder.id == order_id).first()
    if not order:
        return HTMLResponse("工单不存在", status_code=404)
    users = db.query(User).all()
    return templates.TemplateResponse(
        "work_order_form.html",
        {
            "request": request,
            "active": "work_orders",
            "order": order,
            "users": users,
            "user": user,
        },
    )


@app.post("/work-orders/{order_id}/edit")
def update_work_order(
    request: Request,
    order_id: int,
    title: str = Form(),
    location: str = Form(""),
    inspector_id: int = Form(0),
    description: str = Form(""),
    db: Session = Depends(get_db),
):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    order = db.query(WorkOrder).filter(WorkOrder.id == order_id).first()
    if order:
        order.title = title
        order.location = location
        order.inspector_id = inspector_id if inspector_id else None
        order.description = description
        db.commit()
    return RedirectResponse(url=f"/work-orders/{order_id}", status_code=302)


@app.post("/work-orders/{order_id}/status")
def update_work_order_status(
    request: Request, order_id: int, status: str = Form(), db: Session = Depends(get_db)
):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    order = db.query(WorkOrder).filter(WorkOrder.id == order_id).first()
    if order:
        order.status = status
        if status == "completed":
            order.completed_at = datetime.now()
        db.commit()
    return RedirectResponse(url=f"/work-orders/{order_id}", status_code=302)

# ============================================================
# 10. 实时检测 + 截图保存
# ============================================================
@app.get("/work-orders/{order_id}/detect")
def detect_page(request: Request, order_id: int, db: Session = Depends(get_db)):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    order = db.query(WorkOrder).filter(WorkOrder.id == order_id).first()
    if not order:
        return HTMLResponse("工单不存在", status_code=404)
    return templates.TemplateResponse(
        "detect.html",
        {"request": request, "active": "work_orders", "order": order, "user": user},
    )


@app.post("/capture/{work_order_id}")
def capture_snapshot(work_order_id: int, db: Session = Depends(get_db)):
    """从当前视频流截取一帧，运行 YOLO 推理并保存结果"""
    try:
        frame = frame_queue.get(timeout=5)
    except queue.Empty:
        return JSONResponse({"success": False, "error": "无法获取摄像头画面"}, status_code=503)

    # YOLO 推理
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    # 提取检测数据
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        # 没有检测到裂缝，仍然保存原图
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        orig_filename = f"no_crack_{timestamp}.jpg"
        orig_path = os.path.join("images", "originals", orig_filename)
        cv2.imwrite(orig_path, frame)
        return JSONResponse({
            "success": True,
            "has_crack": False,
            "image_url": f"/images/originals/{orig_filename}",
            "message": "未检测到裂缝",
        })

    # 有检测结果，保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    orig_filename = f"{timestamp}.jpg"
    annot_filename = f"{timestamp}_annot.jpg"
    orig_path = os.path.join("images", "originals", orig_filename)
    annot_path = os.path.join("images", "annotated", annot_filename)

    cv2.imwrite(orig_path, frame)
    cv2.imwrite(annot_path, annotated_frame)

    # 汇总检测数据（取置信度最高的裂缝）
    max_conf = float(boxes.conf.max())
    total_area = 0
    total_length = 0.0

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1
        area = w * h
        total_area += area
        # 用对角线近似长度
        total_length += (w**2 + h**2) ** 0.5

    avg_area = total_area // len(boxes)
    # 严重程度判定
    if avg_area < 5000:
        severity = "low"
    elif avg_area < 20000:
        severity = "medium"
    else:
        severity = "high"

    # 写入数据库
    anomaly = Anomaly(
        work_order_id=work_order_id,
        image_path=f"images/originals/{orig_filename}",
        annotated_path=f"images/annotated/{annot_filename}",
        confidence=round(max_conf, 4),
        severity=severity,
        area_pixels=avg_area,
        length_pixels=round(total_length, 2),
    )
    db.add(anomaly)
    db.commit()

    return JSONResponse({
        "success": True,
        "has_crack": True,
        "anomaly_id": anomaly.id,
        "image_url": f"/images/annotated/{annot_filename}",
        "confidence": round(max_conf, 4),
        "severity": severity,
        "count": len(boxes),
    })


# ============================================================
# 11. 工单检测图片列表
# ============================================================
@app.get("/work-orders/{order_id}/images")
def work_order_images(request: Request, order_id: int, db: Session = Depends(get_db)):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    order = db.query(WorkOrder).filter(WorkOrder.id == order_id).first()
    if not order:
        return HTMLResponse("工单不存在", status_code=404)
    anomalies = (
        db.query(Anomaly)
        .filter(Anomaly.work_order_id == order_id)
        .order_by(Anomaly.created_at.desc())
        .all()
    )
    return templates.TemplateResponse(
        "images.html",
        {
            "request": request,
            "active": "work_orders",
            "order": order,
            "anomalies": anomalies,
            "user": user,
        },
    )


# ============================================================
# 12. 静态图片访问
# ============================================================
from fastapi.responses import FileResponse

@app.get("/images/{subdir:path}/{filename}")
def get_image(subdir: str, filename: str):
    file_path = os.path.join("images", subdir, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return HTMLResponse("图片不存在", status_code=404)

# ============================================================
# 13. 历史查询
# ============================================================
@app.get("/history")
def history_page(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(
        "history.html", {"request": request, "active": "history", "user": user}
    )


@app.get("/history/data")
def history_data(
    request: Request,
    page: int = 1,
    date_from: str = "",
    date_to: str = "",
    location: str = "",
    severity: str = "",
    db: Session = Depends(get_db),
):
    user = get_current_user(request)
    if not user:
        return JSONResponse({"error": "未登录"}, status_code=401)

    per_page = 12
    query = (
        db.query(Anomaly, WorkOrder.title, WorkOrder.location)
        .join(WorkOrder, Anomaly.work_order_id == WorkOrder.id)
        .order_by(Anomaly.created_at.desc())
    )

    if severity:
        query = query.filter(Anomaly.severity == severity)
    if location:
        query = query.filter(WorkOrder.location.contains(location))
    if date_from:
        query = query.filter(Anomaly.created_at >= datetime.strptime(date_from, "%Y-%m-%d"))
    if date_to:
        query = query.filter(
            Anomaly.created_at <= datetime.strptime(date_to, "%Y-%m-%d") + timedelta(days=1)
        )

    total = query.count()
    items = query.offset((page - 1) * per_page).limit(per_page).all()

    data = []
    for anomaly, title, loc in items:
        data.append({
            "id": anomaly.id,
            "work_order_title": title,
            "location": loc,
            "image_url": f"/{anomaly.annotated_path}",
            "original_url": f"/{anomaly.image_path}",
            "confidence": anomaly.confidence,
            "severity": anomaly.severity,
            "area_pixels": anomaly.area_pixels,
            "created_at": anomaly.created_at.strftime("%Y-%m-%d %H:%M") if anomaly.created_at else "",
        })

    return {"items": data, "total": total, "page": page, "per_page": per_page}


# ============================================================
# 14. 报告生成
# ============================================================
@app.get("/report/{order_id}")
def report_page(request: Request, order_id: int, db: Session = Depends(get_db)):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    order = db.query(WorkOrder).filter(WorkOrder.id == order_id).first()
    if not order:
        return HTMLResponse("工单不存在", status_code=404)

    inspector = db.query(User).filter(User.id == order.inspector_id).first()
    anomalies = (
        db.query(Anomaly)
        .filter(Anomaly.work_order_id == order_id)
        .order_by(Anomaly.created_at.desc())
        .all()
    )

    # 统计数据
    total_cracks = len(anomalies)
    severity_count = {"low": 0, "medium": 0, "high": 0}
    avg_confidence = 0.0
    if total_cracks > 0:
        avg_confidence = sum(a.confidence for a in anomalies) / total_cracks
        for a in anomalies:
            if a.severity in severity_count:
                severity_count[a.severity] += 1

    return templates.TemplateResponse(
        "report.html",
        {
            "request": request,
            "order": order,
            "inspector": inspector,
            "anomalies": anomalies,
            "total_cracks": total_cracks,
            "severity_count": severity_count,
            "avg_confidence": round(avg_confidence, 4),
            "user": user,
        },
    )
