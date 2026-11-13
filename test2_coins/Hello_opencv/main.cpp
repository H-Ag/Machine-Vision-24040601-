// ============================================================================
//  图像二值化 + 连通域标记（4 邻接，等价表 Two-pass）
//    1. 灰度图 -> 全局大津法二值化
//    2. 手写等价表 Two-pass 标记连通域
//    3. 给每个连通域填不同颜色，并输出统计表 + 中心坐标表 + 三张结果图
// ============================================================================

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>   // cv::utils::logging::setLogLevel

#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

using namespace cv;
using namespace std;

// 面积小于它的连通域当噪点丢掉；设 0 表示不过滤
static const int kMinArea = 200;

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
// 同时把面积小于 minArea 的前景丢掉，并把标签重排成连续的 1..N
// ============================================================================
struct Blob {
    int label, x, y, w, h, area;
    double cx, cy;
};

static vector<Blob> analyze(Mat& labels, int& numLabels, int minArea)
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

    // 过滤小噪点 + 重新编号（背景 0 不动）
    vector<int> remap(n, 0);
    int compact = 1;
    for (int i = 1; i < n; ++i)
        if (area[i] >= minArea) remap[i] = compact++;

    vector<Blob> out;
    for (int i = 0; i < n; ++i) {
        const int lb = (i == 0) ? 0 : remap[i];
        if (i != 0 && lb == 0) continue;                 // 被丢掉的小块
        out.push_back(Blob{ lb, minx[i], miny[i], maxx[i] - minx[i] + 1,
                            maxy[i] - miny[i] + 1, (int)area[i],
                            double(sx[i]) / area[i], double(sy[i]) / area[i] });
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
// 给每个连通域填不同颜色（背景保持黑色）
// ============================================================================
static Mat colorLabels(const Mat& labels)
{
    static const Scalar pal[] = {
        Scalar(  0,   0, 255), Scalar(  0, 165, 255), Scalar(  0, 255, 255),
        Scalar(  0, 255,   0), Scalar(255, 255,   0), Scalar(128, 255, 255),
        Scalar(255,   0,   0), Scalar(255,   0, 255), Scalar(128,   0, 255),
        Scalar(255, 128,   0), Scalar(128, 255,   0), Scalar(  0, 128, 255),
        Scalar(203, 192, 255), Scalar( 60,  20, 220), Scalar( 40, 200, 120),
        Scalar(255,   0, 128), Scalar(128,   0,   0), Scalar(  0,   0, 128),
    };
    const int n = (int)(sizeof(pal) / sizeof(pal[0]));

    Mat out = Mat::zeros(labels.size(), CV_8UC3);
    for (int y = 0; y < labels.rows; ++y) {
        const int* lrow = labels.ptr<int>(y);
        Vec3b* crow = out.ptr<Vec3b>(y);
        for (int x = 0; x < labels.cols; ++x) {
            const int lb = lrow[x];
            if (lb <= 0) continue;                       // 背景保持黑
            const Scalar& c = pal[lb % n];
            crow[x] = Vec3b(uchar(c[0]), uchar(c[1]), uchar(c[2]));
        }
    }
    return out;
}

// ============================================================================
// 画外接矩形 + 序号 + 中心点（背景不画）
// ============================================================================
static void drawBoxes(Mat& canvas, const vector<Blob>& blobs)
{
    for (const Blob& b : blobs) {
        if (b.label == 0) continue;
        rectangle(canvas, Rect(b.x, b.y, b.w, b.h), Scalar(0, 0, 255), 1);
        circle(canvas, Point((int)b.cx, (int)b.cy), 3, Scalar(255, 255, 255), -1);
        putText(canvas, to_string(b.label), Point(b.x, max(b.y - 4, 12)),
                FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 255, 0), 1);
    }
}

int main(int argc, char** argv)
{
    SetConsoleOutputCP(CP_UTF8);
    utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);

    string path = (argc > 1) ? argv[1]
                             : "D:/homework/Machine Vision/test2_coins/coins.jpg";

    Mat gray = imread(path, IMREAD_GRAYSCALE);        // 直接按灰度读入
    if (gray.empty()) { cout << "读不到图片：" << path << endl; return -1; }
    cout << "输入图像：" << path << "  (" << gray.cols << " x " << gray.rows << ")" << endl;

    // ---------- 1. 全局大津法二值化 ----------
    Mat binary;
    const double th = threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
    cout << "二值化：全局大津法，阈值 = " << th << endl;
    cout << "前景像素 = " << countNonZero(binary) << endl;

    // ---------- 2. 连通域标记 + 统计 ----------
    int numLabels = 0;
    Mat labels = twoPass(binary, numLabels);
    const int before = numLabels;
    vector<Blob> blobs = analyze(labels, numLabels, kMinArea);

    const int bg = (blobs.empty() || blobs[0].label != 0) ? 0 : 1;
    const int nFore = (int)blobs.size() - bg;
    cout << "连通域数量（含背景）= " << (int)blobs.size()
         << "，其中背景 " << bg << " 个、前景目标 " << nFore << " 个" << endl;
    if (before - 1 - nFore > 0)
        cout << "（已忽略 " << before - 1 - nFore
             << " 个面积小于 " << kMinArea << " 的连通域）" << endl;

    // ---------- 3. 统计表（背景在第 0 行，序号连续） ----------
    cout << "\n" << setw(5) << "序号" << setw(7) << "x" << setw(7) << "y"
         << setw(7) << "w" << setw(7) << "h" << setw(10) << "面积"
         << setw(11) << "cx" << setw(11) << "cy" << endl;
    for (const Blob& b : blobs) {
        cout << setw(5) << b.label << setw(7) << b.x << setw(7) << b.y
             << setw(7) << b.w << setw(7) << b.h << setw(10) << b.area
             << setw(11) << fixed << setprecision(1) << b.cx
             << setw(11) << b.cy << (b.label == 0 ? "  <- 背景" : "") << endl;
    }

    // ---------- 4. 中心坐标表 ----------
    cout << "\n中心坐标（cx = Σx/面积，cy = Σy/面积）：" << endl;
    cout << setw(5) << "序号" << setw(11) << "cx" << setw(11) << "cy"
         << setw(14) << "取整" << endl;
    for (const Blob& b : blobs)
        cout << setw(5) << b.label
             << setw(11) << fixed << setprecision(2) << b.cx
             << setw(11) << b.cy
             << "    (" << (int)(b.cx + 0.5) << ", " << (int)(b.cy + 0.5) << ")" << endl;

    // ---------- 5. 三张结果图 ----------
    Mat color = colorLabels(labels);                 // 每个连通域一种颜色
    Mat binBgr;
    cvtColor(binary, binBgr, COLOR_GRAY2BGR);

    Mat colorBox = color.clone();
    drawBoxes(colorBox, blobs);                      // 彩色标记 + 框 + 序号

    Mat grayBox;
    cvtColor(gray, grayBox, COLOR_GRAY2BGR);
    drawBoxes(grayBox, blobs);                       // 原灰度图 + 框 + 序号

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
    save("labels_color.png", colorBox);
    save("result.png", grayBox);
    cout << "\n结果图所在目录：" << outDir.string() << endl;

    imshow("1 binary", binBgr);
    imshow("2 labels color", colorBox);
    imshow("3 gray + bbox", grayBox);
    waitKey(0);
    return 0;
}
