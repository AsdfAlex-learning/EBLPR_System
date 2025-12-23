function produce_templates()
    % 创建模板文件夹（使用脚本所在目录的相对路径 char_templates/）
    [script_path, ~, ~] = fileparts(mfilename('fullpath'));
    % 拼接相对路径：脚本目录/char_templates
    template_dir = fullfile(script_path, 'char_templates');
    
    % 检查并创建文件夹
    if ~exist(template_dir, 'dir')
        mkdir(template_dir);
    end
    
    % 字符列表
    chars = {'0','1','2','3','4','5','6','7','8','9',...
             'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',...
             '粤','广','州','佛','山'};
    
    % 模板尺寸（高40，宽20）
    h = 40; w = 20;
    
    for i = 1:length(chars)
        % 1. 创建空白二值图像（0=黑，1=白）
        char_bin = zeros(h, w);
        
        % 2. 直接生成字符的二值矩阵（以“3”为例，手动对齐坐标）
        % 注：可根据字符形状调整像素点，以下是通用适配方式
        % 用MATLAB的“textscan”+“imshow”直接渲染标准字符
        % 打开临时图窗，强制渲染为矢量字符
        fig = figure('Units', 'Pixels', 'Position', [0 0 100 200], 'Visible', 'off');
        axes('Position', [0 0 1 1], 'XLim', [0 1], 'YLim', [0 1], 'XTick', [], 'YTick', []);
        text(0.5, 0.5, chars{i}, 'FontSize', 100, 'FontName', 'SimHei', ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
        % 捕获并缩放到目标尺寸
        frame = getframe(fig);
        char_img = frame2im(frame);
        char_img = imresize(char_img, [h w]);
        char_bin = imbinarize(rgb2gray(char_img), 0.5); % 二值化（0=黑，1=白）
        close(fig);
        
        % 3. 转成黑底白字（匹配车牌识别的模板要求）
        char_bin = 1 - char_bin; % 反转：1=白字符，0=黑背景
        char_uint8 = uint8(char_bin * 255); % 转8位格式
        
        % 4. 保存模板（使用拼接好的相对路径）
        save_path = fullfile(template_dir, [chars{i} '.png']);
        imwrite(char_uint8, save_path);
        fprintf('已生成标准模板：%s\n', save_path);
    end
    
    fprintf('所有标准模板生成完成！模板保存路径：%s\n', template_dir);
end