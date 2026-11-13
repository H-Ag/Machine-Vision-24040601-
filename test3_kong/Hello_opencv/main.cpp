// ============================================================================
//  第三题：圆孔检测（图像二值化 + 连通域标记 + 形状筛选）
//    1. 灰度图 -> 反向大津法二值化（孔比金属暗，故用 THRESH_BINARY_INV）
//    2. 手写等价表 Two-pass 标记连通域（4 邻接）
//    3. 按面积 / 填充率 / 宽高比筛出圆形孔洞
//    4. 输出统计表 + 中心坐标表 + 三张结果图
// ============================================================================

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>   // cv::utils::logging::setLogLevel

#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

using namespace cv;
using namespace std;

// ----------------------------------------------------------------------------
// 可调参数
// ----------------------------------------------------------------------------

// 闭运算的核尺寸（奇数像素）。设 1 表示不做闭运算
static const int kCloseSize = 3;

// 圆孔的面积范围（像素）。太小是刻字/划痕，太大是齿轮暗区或中心大孔
static const int kMinArea = 1000;
static const int kMaxArea = 8000;

// 形状约束：填充率 = 面积/外接矩形面积，正圆约 0.785
static const double kMinFill = 0.60;
static const double kMaxFill = 0.88;

// 宽高比 w/h 与 1 的最大偏差（正圆应接近 1）
static const double kMaxAspectDelta = 0.20;

// ============================================================================
// 等价表 Two-pass 标记，4 邻接
//   规则：邻域(上、左)全 0 -> 新标签；只有一种 -> 沿用；两种以上 -> 取最小并记等价
//   等价表用并查集（父指针 + 路径压缩）实现
// ============================================================================
static int dsFind(vector<int>& p, int x)
{
    while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; }
    return x;
}

static Mat twoPass(const Mat& bin, int& numLabels)
{
    const int h = bin.rows, w = bin.cols;
    Mat labels = Mat::zeros(h, w, CV_32S);
    vector<int> parent(1, 0);               // parent[0] 占位
    int next = 1;

    for (int y = 0; y < h; ++y) {
        const uchar* brow = bin.ptr<uchar>(y);
        int* lrow = labels.ptr<int>(y);
        const int* lup = (y > 0) ? labels.ptr<int>(y - 1) : nullptr;

        for (int x = 0; x < w; ++x) {
            if (brow[x] == 0) continue;
            const int up = lup ? lup[x] : 0;
            const int left = (x > 0) ? lrow[x - 1] : 0;

            if (up == 0 && left == 0) {                  // a. 新标签
                parent.push_back(next);
                lrow[x] = next++;
            } else if (up == 0 || left == 0) {            // b. 沿用唯一标签
                lrow[x] = up ? up : left;
            } else {                                     // c. 取最小值并记等价
                lrow[x] = min(up, left);
                if (up != left) {
                    const int a = dsFind(parent, up), b = dsFind(parent, left);
                    if (a != b) parent[max(a, b)] = min(a, b);   // 根始终是集合最小值
                }
            }
        }
    }

    // 第二次遍历：改赋等价根，并压成连续编号 1..N
    vector<int> remap(parent.size(), 0);
    int compact = 1;
    for (int y = 0; y < h; ++y) {
        int* lrow = labels.ptr<int>(y);
        for (int x = 0; x < w; ++x) {
            const int lb = lrow[x];
            if (lb == 0) continue;
            const int root = dsFind(parent, lb);
            if (remap[root] == 0) remap[root] = compact++;
            lrow[x] = remap[root];
        }
    }
    numLabels = compact;
    return labels;
}

// ============================================================================
// 统计每个连通域：外接矩形、面积、中心坐标（0 号是背景，也统计）
// 同时按「面积 + 填充率 + 宽高比」筛出圆形孔洞，并把标签重排成连续的 1..N
// ============================================================================
struct Blob {
    int label, x, y, w, h, area;
    double cx, cy, fill;
};

static vector<Blob> analyze(Mat& labels, int& numLabels)
{
    const int h = labels.rows, w = labels.cols, n = numLabels;

    vector<int> minx(n, w), miny(n, h), maxx(n, -1), maxy(n, -1);
    vector<long long> sx(n, 0), sy(n, 0), area(n, 0);

    for (int y = 0; y < h; ++y) {
        const int* lrow = labels.ptr<int>(y);
        for (int x = 0; x < w; ++x) {
            const int lb = lrow[x];
            if (x < minx[lb]) minx[lb] = x;
            if (x > maxx[lb]) maxx[lb] = x;
            if (y < miny[lb]) miny[lb] = y;
            if (y > maxy[lb]) maxy[lb] = y;
            sx[lb] += x;
            sy[lb] += y;
            area[lb] += 1;
        }
    }

    // 逐个判断是不是圆孔：面积在范围内、不贴边、形状接近圆
    vector<int> remap(n, 0);
    int compact = 1;
    for (int i = 1; i < n; ++i) {
        const int bw = maxx[i] - minx[i] + 1;
        const int bh = maxy[i] - miny[i] + 1;
        const bool border = (minx[i] == 0) || (miny[i] == 0) ||
                            (maxx[i] == w - 1) || (maxy[i] == h - 1);
        if (area[i] < kMinArea || area[i] > kMaxArea || border) continue;

        const double fill = double(area[i]) / (double(bw) * bh);
        const double aspect = double(bw) / double(bh);
        if (fill < kMinFill || fill > kMaxFill) continue;
        if (fabs(aspect - 1.0) > kMaxAspectDelta) continue;

        remap[i] = compact++;
    }

    vector<Blob> out;
    for (int i = 0; i < n; ++i) {
        const int lb = (i == 0) ? 0 : remap[i];
        if (i != 0 && lb == 0) continue;                 // 背景或被筛掉的区域
        const int bw = maxx[i] - minx[i] + 1;
        const int bh = maxy[i] - miny[i] + 1;
        out.push_back(Blob{ lb, minx[i], miny[i], bw, bh, (int)area[i],
                            double(sx[i]) / area[i], double(sy[i]) / area[i],
                            double(area[i]) / (double(bw) * bh) });
    }

    // 按新编号写回标签图，保证图上的序号和表格一致
    for (int y = 0; y < h; ++y) {
        int* lrow = labels.ptr<int>(y);
        for (int x = 0; x < w; ++x)
            if (lrow[x] > 0) lrow[x] = remap[lrow[x]];
    }
    numLabels = compact;
    return out;
}

// ============================================================================
// 给每个圆孔填不同颜色（其余保持黑色）
// ============================================================================
static Mat colorLabels(const Mat& labels)
{
    static const Scalar pal[] = {
        Scalar(  0,   0, 255), Scalar(  0, 165, 255), Scalar(  0, 255, 255),
        Scalar(  0, 255,   0), Scalar(255, 255,   0), Scalar(128, 255, 255),
        Scalar(255,   0,   0), Scalar(255,   0, 255), Scalar(128,   0, 255),
        Scalar(255, 128,   0), Scalar(128, 255,   0), Scalar(  0, 128, 255),
    };
    const int n = (int)(sizeof(pal) / sizeof(pal[0]));

    Mat out = Mat::zeros(labels.size(), CV_8UC3);
    for (int y = 0; y < labels.rows; ++y) {
        const int* lrow = labels.ptr<int>(y);
        Vec3b* crow = out.ptr<Vec3b>(y);
        for (int x = 0; x < labels.cols; ++x) {
            const int lb = lrow[x];
            if (lb <= 0) continue;
            const Scalar& c = pal[lb % n];
            crow[x] = Vec3b(uchar(c[0]), uchar(c[1]), uchar(c[2]));
        }
    }
    return out;
}

// ============================================================================
// 画圆：圆心取连通域质心，半径取 (w+h)/4 —— 拟合外接圆
// ============================================================================
static void drawCircles(Mat& canvas, const vector<Blob>& blobs)
{
    for (const Blob& b : blobs) {
        if (b.label == 0) continue;
        const Point c((int)(b.cx + 0.5), (int)(b.cy + 0.5));
        const int r = (b.w + b.h) / 4;
        circle(canvas, c, r, Scalar(0, 0, 255), 2);            // 红色外接圆
        line(canvas, Point(c.x - 5, c.y), Point(c.x + 5, c.y), Scalar(0, 255, 0), 1);
        line(canvas, Point(c.x, c.y - 5), Point(c.x, c.y + 5), Scalar(0, 255, 0), 1);
        putText(canvas, to_string(b.label), Point(max(c.x - r, 2), max(c.y - r - 8, 16)),
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
    }
}

int main(int argc, char** argv)
{
    SetConsoleOutputCP(CP_UTF8);
    utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);

    string path = (argc > 1) ? argv[1]
                             : "D:/homework/Machine Vision/test3_kong/image.png";

    Mat gray = imread(path, IMREAD_GRAYSCALE);        // 直接按灰度读入
    if (gray.empty()) { cout << "读不到图片：" << path << endl; return -1; }
    cout << "输入图像：" << path << "  (" << gray.cols << " x " << gray.rows << ")" << endl;

    // ---------- 1. 反向大津法二值化：孔比周围金属暗 ----------
    Mat binary;
    const double th = threshold(gray, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
    cout << "二值化：全局大津法（反向，暗目标 = 孔），阈值 = " << th << endl;
    cout << "二值化后前景像素 = " << countNonZero(binary) << endl;

    // ---------- 1.5 闭运算：填掉孔内小缺口、去掉孤立噪点 ----------
    if (kCloseSize > 1) {
        const Mat kernel = getStructuringElement(
            MORPH_ELLIPSE, Size(kCloseSize, kCloseSize));
        morphologyEx(binary, binary, MORPH_CLOSE, kernel);
        cout << "闭运算：椭圆核 " << kCloseSize << "x" << kCloseSize
             << "，前景像素 = " << countNonZero(binary) << endl;
    }

    // ---------- 2. 连通域标记 + 形状筛选 ----------
    int numLabels = 0;
    Mat labels = twoPass(binary, numLabels);
    const int before = numLabels;
    vector<Blob> blobs = analyze(labels, numLabels);

    const int bg = (blobs.empty() || blobs[0].label != 0) ? 0 : 1;
    const int nHole = (int)blobs.size() - bg;
    cout << "连通域总数（含背景）= " << before << endl;
    cout << "检测到圆孔 = " << nHole << " 个（筛掉 " << (before - 1 - nHole)
         << " 个不符合面积/形状条件的区域）" << endl;

    // ---------- 3. 统计表 ----------
    cout << "\n" << setw(5) << "序号" << setw(7) << "x" << setw(7) << "y"
         << setw(7) << "w" << setw(7) << "h" << setw(11) << "面积"
         << setw(12) << "填充率" << setw(9) << "直径"
         << setw(10) << "cx" << setw(10) << "cy" << endl;
    for (const Blob& b : blobs) {
        if (b.label == 0) continue;                      // 背景行不列出
        cout << setw(5) << b.label << setw(7) << b.x << setw(7) << b.y
             << setw(7) << b.w << setw(7) << b.h << setw(9) << b.area
             << setw(9) << fixed << setprecision(3) << b.fill
             << setw(9) << setprecision(1) << double(b.w + b.h) / 2
             << setw(10) << b.cx << setw(10) << b.cy << endl;
    }

    // ---------- 4. 中心坐标表 ----------
    cout << "\n圆孔中心坐标（cx = Σx/面积，cy = Σy/面积）：" << endl;
    cout << setw(5) << "序号" << setw(11) << "cx" << setw(11) << "cy"
         << setw(14) << "取整" << endl;
    for (const Blob& b : blobs) {
        if (b.label == 0) continue;
        cout << setw(5) << b.label
             << setw(11) << fixed << setprecision(2) << b.cx
             << setw(11) << b.cy
             << "    (" << (int)(b.cx + 0.5) << ", " << (int)(b.cy + 0.5) << ")" << endl;
    }

    // ---------- 5. 三张结果图 ----------
    Mat color = colorLabels(labels);                     // 每个圆孔一种颜色
    Mat binBgr;
    cvtColor(binary, binBgr, COLOR_GRAY2BGR);

    Mat colorCircle = color.clone();
    drawCircles(colorCircle, blobs);                     // 彩色标记 + 外接圆 + 序号

    Mat grayCircle;
    cvtColor(gray, grayCircle, COLOR_GRAY2BGR);
    drawCircles(grayCircle, blobs);                      // 原灰度图 + 外接圆 + 序号

    // 输出目录：不存在就自动创建（相对当前工作目录，VS 里即工程目录）
    std::error_code ec;
    std::filesystem::create_directories("output", ec);
    const std::filesystem::path outDir = std::filesystem::absolute("output");

    auto save = [&](const std::string& name, const Mat& img) {
        const std::filesystem::path f = outDir / name;
        if (!imwrite(f.string(), img)) cout << "保存失败：" << f.string() << endl;
        else                          cout << "已保存：" << f.string() << endl;
    };

    save("binary.png", binBgr);
    save("labels_color.png", colorCircle);
    save("result.png", grayCircle);
    cout << "\n结果图所在目录：" << outDir.string() << endl;

    imshow("1 binary", binBgr);
    imshow("2 labels color", colorCircle);
    imshow("3 gray + circles", grayCircle);
    waitKey(0);
    return 0;
}
