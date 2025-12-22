function [plate_path, plate_number] = process_image(imagePath)
% PROCESS_IMAGE 车牌预处理+字符识别主函数
%   输入参数:
%       imagePath - 原始图像文件路径（字符串）
%   输出参数:
%       plate_path - 裁剪后的车牌图像文件路径（字符串）
%       plate_number - 识别出的车牌号码（字符串）
    if ~exist(imagePath, 'file')
        error('图像文件不存在: %s', imagePath);
    end
    
    img = imread(imagePath);
    if size(img, 3) == 3
        img_gray = rgb2gray(img);
    else
        img_gray = img;
    end
    img_processed = preprocess_image(img_gray);
    plate_region = locate_plate_region(img_processed);
    bbox = plate_region.BoundingBox;
    x1 = max(1, round(bbox(1)));
    y1 = max(1, round(bbox(2)));
    x2 = min(size(img_gray, 2), round(bbox(1) + bbox(3)));
    y2 = min(size(img_gray, 1), round(bbox(2) + bbox(4)));

    if x2 <= x1 || y2 <= y1
        plate_img = img_gray;
    else
        plate_img = img_gray(y1:y2, x1:x2);
    end

    [folder, name, ~] = fileparts(imagePath);
    if isempty(folder)
        folder = pwd;
    end
    plate_path = fullfile(folder, [name, '_plate.png']);
    imwrite(plate_img, plate_path);
    fprintf('MATLAB 预处理完成，车牌子图保存到: %s\n', plate_path);

    % ========== 新增：直接在MATLAB中完成字符识别 ==========
    % 检测图像类型并选择对应算法
    img_type = detect_image_type(img_processed);
    switch img_type
        case 'normal'
            plate_number = recognize_normal_plate(img_processed);
        case 'tilted'
            plate_number = recognize_tilted_plate(img_processed);
        case 'interfered'
            plate_number = recognize_interfered_plate(img_processed);
        case 'multiple'
            plate_number = recognize_multiple_plates(img_processed);
        otherwise
            plate_number = '识别失败：未知图像类型';
    end
    fprintf('MATLAB 字符识别完成，车牌号码：%s\n', plate_number);
end

%% ==================== 辅助函数 ====================

function img_processed = preprocess_image(img)
% 图像预处理：增强对比度、去噪、二值化
    
    % 1. 直方图均衡化增强对比度
    img_eq = adapthisteq(img);
    
    % 2. 中值滤波去噪
    img_filtered = medfilt2(img_eq, [3, 3]);
    
    % 3. 自适应二值化
    img_binary = imbinarize(img_filtered, 'adaptive', 'Sensitivity', 0.5);
    
    % 4. 形态学操作：去除小噪点
    se = strel('disk', 2);
    img_processed = imopen(img_binary, se);
    img_processed = imclose(img_processed, se);
end

function img_type = detect_image_type(img)
% 检测图像类型：正常、倾斜、干扰、多车牌
    
    % 检测倾斜角度
    angle = detect_tilt_angle(img);
    
    % 检测是否有多个车牌区域
    num_plates = detect_plate_count(img);
    
    % 检测干扰程度
    interference_level = detect_interference(img);
    
    % 根据特征判断图像类型
    if num_plates > 1
        img_type = 'multiple';
    elseif abs(angle) > 2 && abs(angle) <= 15
        img_type = 'tilted';
    elseif interference_level > 0.3
        img_type = 'interfered';
    else
        img_type = 'normal';
    end
end

function angle = detect_tilt_angle(img)
% 检测图像倾斜角度（使用Hough变换）
    
    % 边缘检测
    edges = edge(img, 'Canny');
    
    % Hough变换检测直线
    [H, theta, rho] = hough(edges);
    peaks = houghpeaks(H, 5, 'threshold', ceil(0.3 * max(H(:))));
    
    if ~isempty(peaks)
        % 计算主要直线的角度
        angles = theta(peaks(:, 2));
        % 过滤接近水平或垂直的线（±5度）
        valid_angles = angles(abs(angles) > 5 & abs(angles) < 85);
        if ~isempty(valid_angles)
            angle = mean(valid_angles);
        else
            angle = 0;
        end
    else
        angle = 0;
    end
end

function count = detect_plate_count(img)
% 检测图像中车牌区域的数量
    
    % 使用连通域分析
    cc = bwconncomp(~img, 8);
    
    % 计算每个连通域的面积
    stats = regionprops(cc, 'Area', 'BoundingBox');
    areas = [stats.Area];
    
    % 过滤小区域，保留可能是车牌的区域
    % 假设车牌区域面积在图像总面积的5%-50%之间
    img_area = size(img, 1) * size(img, 2);
    min_area = img_area * 0.05;
    max_area = img_area * 0.5;
    
    valid_regions = areas >= min_area & areas <= max_area;
    count = sum(valid_regions);
    
    % 限制最大数量
    count = min(count, 5);
end

function level = detect_interference(img)
% 检测图像中的干扰程度
    
    % 计算图像的复杂度（使用梯度）
    [gx, gy] = gradient(double(img));
    gradient_magnitude = sqrt(gx.^2 + gy.^2);
    
    % 计算梯度方差（干扰越多，梯度变化越大）
    level = std(gradient_magnitude(:)) / max(gradient_magnitude(:));
    
    % 归一化到0-1范围
    level = min(level, 1);
end

%% ==================== 算法1: 正常车牌识别 ====================

function result = recognize_normal_plate(img)
% 正常车牌识别算法：对清晰、无干扰的电动自行车车牌进行字符识别
    
    fprintf('使用算法: 正常车牌识别\n');
    
    % 1. 进一步预处理
    img_clean = enhance_normal_plate(img);
    
    % 2. 车牌区域定位
    plate_region = locate_plate_region(img_clean);
    
    % 3. 提取车牌区域图像
    bbox = plate_region.BoundingBox;
    x1 = max(1, round(bbox(1)));
    y1 = max(1, round(bbox(2)));
    x2 = min(size(img_clean, 2), round(bbox(1) + bbox(3)));
    y2 = min(size(img_clean, 1), round(bbox(2) + bbox(4)));
    plate_img = img_clean(y1:y2, x1:x2);
    
    % 4. 字符分割
    plate_region_img = struct('BoundingBox', [1, 1, size(plate_img, 2), size(plate_img, 1)], 'Image', plate_img);
    characters = segment_characters(plate_region_img);
    
    % 5. 字符识别
    result = recognize_characters(characters);
    
    if isempty(result)
        result = '识别失败';
    end
end

function img_enhanced = enhance_normal_plate(img)
% 增强正常车牌图像
    
    % 对比度增强
    img_enhanced = imadjust(img);
    
    % 轻微锐化
    h = fspecial('unsharp', 0.5);
    img_enhanced = imfilter(img_enhanced, h, 'replicate');
end

%% ==================== 算法2: 倾斜车牌矫正 ====================

function result = recognize_tilted_plate(img)
% 倾斜车牌矫正算法：对倾斜角度≤15°的车牌进行矫正后识别
    
    fprintf('使用算法: 倾斜车牌矫正识别\n');
    
    % 1. 检测倾斜角度
    angle = detect_tilt_angle(img);
    fprintf('检测到倾斜角度: %.2f度\n', angle);
    
    % 限制角度范围
    if abs(angle) > 15
        angle = sign(angle) * 15;  % 限制在±15度内
    end
    
    % 2. 图像旋转矫正
    img_corrected = imrotate(img, -angle, 'bilinear', 'crop');
    
    % 3. 重新预处理
    img_processed = preprocess_image(img_corrected);
    
    % 4. 使用正常识别流程
    result = recognize_normal_plate(img_processed);
end

%% ==================== 算法3: 文字干扰车牌处理 ====================

function result = recognize_interfered_plate(img)
% 文字干扰车牌处理算法：对存在少量文字/图案干扰的车牌进行去干扰后识别
    
    fprintf('使用算法: 文字干扰车牌处理\n');
    
    % 1. 去除干扰
    img_cleaned = remove_interference(img);
    
    % 2. 增强图像
    img_enhanced = enhance_after_cleaning(img_cleaned);
    
    % 3. 使用正常识别流程
    result = recognize_normal_plate(img_enhanced);
end

function img_cleaned = remove_interference(img)
% 去除图像中的干扰文字和图案
    
    % 1. 形态学操作去除小干扰
    se_small = strel('disk', 1);
    img_cleaned = imopen(img, se_small);
    
    % 2. 使用连通域分析去除孤立的小区域
    cc = bwconncomp(~img_cleaned, 8);
    stats = regionprops(cc, 'Area');
    
    % 计算面积阈值（小于平均面积的区域可能是干扰）
    if cc.NumObjects > 0
        areas = [stats.Area];
        mean_area = mean(areas);
        min_area = mean_area * 0.1;  % 保留大于10%平均面积的区域
        
        % 创建掩码
        mask = false(size(img_cleaned));
        for i = 1:cc.NumObjects
            if stats(i).Area >= min_area
                mask(cc.PixelIdxList{i}) = true;
            end
        end
        img_cleaned = img_cleaned | ~mask;
    end
    
    % 3. 中值滤波进一步平滑（如果是逻辑图像，先转换）
    if islogical(img_cleaned)
        img_cleaned = medfilt2(double(img_cleaned), [3, 3]) > 0.5;
    else
        img_cleaned = medfilt2(img_cleaned, [3, 3]);
    end
end

function img_enhanced = enhance_after_cleaning(img)
% 去干扰后的图像增强
    
    % 对比度增强
    img_enhanced = imadjust(img);
    
    % 形态学闭运算连接字符
    se = strel('rectangle', [2, 1]);
    img_enhanced = imclose(img_enhanced, se);
end

%% ==================== 算法4: 多车牌场景识别 ====================

function result = recognize_multiple_plates(img)
% 多车牌场景识别算法：从包含多个车牌的图像中识别目标车牌
    
    fprintf('使用算法: 多车牌场景识别\n');
    
    % 1. 检测所有车牌区域
    plate_regions = detect_all_plates(img);
    
    if isempty(plate_regions)
        result = '未检测到车牌';
        return;
    end
    
    fprintf('检测到 %d 个车牌区域\n', length(plate_regions));
    
    % 2. 选择目标车牌（选择最大或最清晰的车牌）
    target_plate = select_target_plate(plate_regions, img);
    
    % 3. 提取目标车牌区域
    img_plate = extract_plate_region(img, target_plate);
    
    % 4. 对目标车牌进行识别
    result = recognize_normal_plate(img_plate);
end

function plate_regions = detect_all_plates(img)
% 检测图像中所有可能的车牌区域
    
    % 使用连通域分析
    cc = bwconncomp(~img, 8);
    stats = regionprops(cc, 'Area', 'BoundingBox', 'Extent', 'Eccentricity');
    
    % 计算图像总面积
    img_area = size(img, 1) * size(img, 2);
    
    plate_regions = struct('BoundingBox', {}, 'Area', {});
    region_count = 0;
    
    for i = 1:cc.NumObjects
        % 过滤条件：面积、宽高比、紧凑度
        area = stats(i).Area;
        bbox = stats(i).BoundingBox;
        width = bbox(3);
        height = bbox(4);
        
        if height > 0
            aspect_ratio = width / height;
            extent = stats(i).Extent;
            
            % 车牌特征：面积在5%-50%之间，宽高比约1.5-6，紧凑度>0.5
            if area >= img_area * 0.05 && area <= img_area * 0.5 && ...
               aspect_ratio >= 1.5 && aspect_ratio <= 6 && ...
               extent > 0.5
                region_count = region_count + 1;
                plate_regions(region_count).BoundingBox = bbox;
                plate_regions(region_count).Area = area;
            end
        end
    end
    
    % 按面积排序
    if region_count > 0
        [~, idx] = sort([plate_regions.Area], 'descend');
        plate_regions = plate_regions(idx);
    end
end

function target_plate = select_target_plate(plate_regions, img)
% 选择目标车牌（选择最大且最清晰的车牌）
    
    if isempty(plate_regions)
        error('未找到车牌区域');
    end
    
    if length(plate_regions) == 1
        target_plate = plate_regions(1);
        return;
    end
    
    % 计算每个区域的清晰度（使用梯度）
    scores = zeros(length(plate_regions), 1);
    
    for i = 1:length(plate_regions)
        bbox = plate_regions(i).BoundingBox;
        x1 = max(1, round(bbox(1)));
        y1 = max(1, round(bbox(2)));
        x2 = min(size(img, 2), round(bbox(1) + bbox(3)));
        y2 = min(size(img, 1), round(bbox(2) + bbox(4)));
        
        if x2 > x1 && y2 > y1
            region = img(y1:y2, x1:x2);
            
            % 计算清晰度（梯度方差）
            if islogical(region)
                region = double(region);
            end
            [gx, gy] = gradient(region);
            gradient_mag = sqrt(gx.^2 + gy.^2);
            
            if ~isempty(gradient_mag(:))
                scores(i) = std(gradient_mag(:)) * plate_regions(i).Area;  % 清晰度 × 面积
            else
                scores(i) = plate_regions(i).Area;  % 如果无法计算梯度，仅使用面积
            end
        else
            scores(i) = plate_regions(i).Area;  % 无效区域，仅使用面积
        end
    end
    
    % 选择得分最高的
    [~, idx] = max(scores);
    target_plate = plate_regions(idx);
end

function img_plate = extract_plate_region(img, plate_region)
% 提取车牌区域
    
    bbox = plate_region.BoundingBox;
    x1 = max(1, round(bbox(1)));
    y1 = max(1, round(bbox(2)));
    x2 = min(size(img, 2), round(bbox(1) + bbox(3)));
    y2 = min(size(img, 1), round(bbox(2) + bbox(4)));
    
    if x2 > x1 && y2 > y1
        img_plate = img(y1:y2, x1:x2);
    else
        % 如果区域无效，返回整个图像
        img_plate = img;
    end
end

%% ==================== 通用识别函数 ====================

function plate_region = locate_plate_region(img)
% 定位车牌区域
    
    % 使用连通域分析
    cc = bwconncomp(~img, 8);
    stats = regionprops(cc, 'Area', 'BoundingBox', 'Extent');
    
    img_area = size(img, 1) * size(img, 2);
    best_score = 0;
    best_idx = 0;
    
    for i = 1:cc.NumObjects
        area = stats(i).Area;
        bbox = stats(i).BoundingBox;
        width = bbox(3);
        height = bbox(4);
        
        if height > 0
            aspect_ratio = width / height;
            extent = stats(i).Extent;
            
            % 评分：面积适中、宽高比合理、紧凑度高
            if area >= img_area * 0.1 && area <= img_area * 0.8 && ...
               aspect_ratio >= 2 && aspect_ratio <= 5
                score = area * extent * (1 / (abs(aspect_ratio - 3) + 0.1));  % 偏好宽高比接近3
                if score > best_score
                    best_score = score;
                    best_idx = i;
                end
            end
        end
    end
    
    if best_idx > 0 && best_idx <= length(stats)
        bbox = stats(best_idx).BoundingBox;
        plate_region = struct('BoundingBox', bbox, 'Image', img);
    else
        % 如果没找到合适的区域，返回整个图像
        plate_region = struct('BoundingBox', [1, 1, size(img, 2), size(img, 1)], 'Image', img);
    end
end

function characters = segment_characters(plate_region)
% 字符分割：将车牌区域分割成单个字符
    
    % 提取车牌区域图像
    bbox = plate_region.BoundingBox;
    x1 = max(1, round(bbox(1)));
    y1 = max(1, round(bbox(2)));
    x2 = min(size(plate_region.Image, 2), round(bbox(1) + bbox(3)));
    y2 = min(size(plate_region.Image, 1), round(bbox(2) + bbox(4)));
    
    plate_img = plate_region.Image(y1:y2, x1:x2);
    
    % 垂直投影分割字符
    vertical_projection = sum(~plate_img, 1);  % 每列的白色像素数
    
    % 找到字符边界
    threshold = max(vertical_projection) * 0.1;  % 阈值
    char_boundaries = find(vertical_projection > threshold);
    
    if isempty(char_boundaries)
        % 如果分割失败，返回整个车牌
        characters(1).image = plate_img;
        characters(1).position = [1, 1];
        return;
    end
    
    % 找到连续区域（字符）
    diff_boundaries = diff([0, char_boundaries, length(vertical_projection) + 1]);
    gaps = find(diff_boundaries > 1);
    
    characters = struct('image', {}, 'position', {});
    char_idx = 1;
    
    for i = 1:length(gaps)
        if i == 1
            start_col = 1;
        else
            start_col = char_boundaries(gaps(i-1));
        end
        
        if i == length(gaps)
            end_col = size(plate_img, 2);
        else
            end_col = char_boundaries(gaps(i));
        end
        
        % 提取字符图像
        char_img = plate_img(:, start_col:end_col);
        
        % 去除上下空白
        horizontal_proj = sum(~char_img, 2);
        rows = find(horizontal_proj > 0);
        if ~isempty(rows)
            char_img = char_img(min(rows):max(rows), :);
        end
        
        if size(char_img, 2) > 5 && size(char_img, 1) > 5  % 过滤太小的区域
            characters(char_idx).image = char_img;
            characters(char_idx).position = [start_col, 1];
            char_idx = char_idx + 1;
        end
    end
    
    if isempty(characters)
        % 如果分割失败，返回整个车牌
        characters(1).image = plate_img;
        characters(1).position = [1, 1];
    end
end

function result = recognize_characters(characters)
% 字符识别：使用模板匹配替代OCR工具箱，识别车牌中的数字/字母
    
    if isempty(characters)
        result = '识别失败：未检测到字符';
        return;
    end
    
    % 初始化结果
    result_text = '';
    
    % 逐个识别字符
    for i = 1:length(characters)
        char_img = characters(i).image;
        % 调用模板匹配函数识别单个字符
        char_label = match_character(char_img);
        result_text = [result_text, char_label];
    end
    
    % 输出识别结果
    if isempty(result_text) || all(result_text == '?')
        result = '识别失败：未匹配到有效字符';
    else
        result = result_text;
    end
end

function load_char_templates()
% 加载字符模板库（全局变量）
% 提前在MATLAB工作目录下创建char_templates文件夹，放入0-9、A-Z的二值化模板图片
% 模板命名规则：0.png, 1.png, ..., 9.png, A.png, B.png, ..., Z.png

    global char_templates char_labels;
    char_templates = [];
    char_labels = [];
    
    % 模板文件夹路径（请根据你的实际路径修改）
    template_dir = fullfile(pwd, 'char_templates');
    if ~exist(template_dir, 'dir')
        error('字符模板文件夹不存在: %s，请创建并放入0-9、A-Z的模板图片', template_dir);
    end
    
    % 定义字符标签顺序
    labels = [{'0'},{'1'},{'2'},{'3'},{'4'},{'5'},{'6'},{'7'},{'8'},{'9'},...
              {'A'},{'B'},{'C'},{'D'},{'E'},{'F'},{'G'},{'H'},{'I'},{'J'},...
              {'K'},{'L'},{'M'},{'N'},{'O'},{'P'},{'Q'},{'R'},{'S'},{'T'},...
              {'U'},{'V'},{'W'},{'X'},{'Y'},{'Z'},{'粤'}];
    
    % 加载每个字符的模板
    for i = 1:length(labels)
        template_path = fullfile(template_dir, [labels{i} '.png']);
        if exist(template_path, 'file')
            % 读取并预处理模板（二值化、归一化尺寸为40x20）
            template = imread(template_path);
            if size(template, 3) == 3
                template = rgb2gray(template);
            end
            template = imbinarize(template);  % 二值化
            template = imresize(template, [40, 20]);  % 统一尺寸（高40，宽20）
            char_templates = cat(3, char_templates, template);
            char_labels = [char_labels, labels{i}];
        else
            warning('模板文件缺失: %s', template_path);
        end
    end
    
    if isempty(char_templates)
        error('未加载到任何字符模板，请检查模板文件夹');
    end
end

function char_label = match_character(char_img)
% 单个字符匹配：计算待识别字符与模板的相似度，返回最匹配的字符
    global char_templates char_labels;
    
    % 初始化模板库（首次调用时加载）
    if isempty(char_templates)
        load_char_templates();
    end
    
    % 预处理待识别字符（统一尺寸、二值化）
    char_img = imbinarize(char_img);  % 二值化
    char_img = imresize(char_img, [40, 20]);  % 与模板尺寸一致
    
    % 计算与每个模板的相似度（归一化互相关）
    max_corr = 0;
    char_label = '';
    num_templates = size(char_templates, 3);
    
    for i = 1:num_templates
        template = char_templates(:, :, i);
        % 计算互相关系数（值越大，相似度越高）
        corr_map = normxcorr2(template, char_img);
        current_corr = max(corr_map(:));
        
        % 更新最佳匹配
        if current_corr > max_corr
            max_corr = current_corr;
            char_label = char_labels{i};
        end
    end
    
    % 相似度阈值过滤（低于0.6视为匹配失败）
    if max_corr < 0.6
        char_label = '?';
    end
end