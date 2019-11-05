import random

import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageColor

EMPTY = "NULL"


def show_img(img):
    cv2.imshow('name', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class Line(object):
    def __init__(self, line_thickness=0, type=1):
        assert line_thickness % (type + 1) == 0
        self.line_thichness = line_thickness
        self.type = type  # 0 là ẩn, n = n vạch

    def set_invisible(self):
        self.type = 0


class Row(object):
    def __init__(self, height, margin_top, margin_bottom, index, line_top: Line, line_bottom: Line):
        self.height = height
        self.index = index
        self.margin_top = margin_top
        self.margin_bottom = margin_bottom
        self.line_top = line_top
        self.line_bottom = line_bottom

    def get_height(self):
        return self.height + self.line_top.line_thichness + self.line_bottom.line_thichness


class Col(object):
    def __init__(self, width, margin_left, margin_right, index, line_left: Line, line_right: Line):
        self.width = width
        self.index = index
        self.margin_left = margin_left
        self.margin_right = margin_right
        self.line_left = line_left
        self.line_right = line_right

    def get_width(self):
        return self.width + self.line_left.line_thichness + self.line_right.line_thichness


class Cell(object):
    def __init__(self, col: Col, row: Row, text: str, cell_id: int):
        self.col = col
        self.row = row
        self.text = text
        self.cell_id = cell_id


class Table(object):
    def __init__(self, widths: list = [900, 300, 300], heights: list = [100, 50, 50, 50, 50, 50, 50, 50],
                 margin_left=5, margin_right=5, margin_top=5, margin_bottom=5):
        self.width_each_cell = widths
        self.heigh_each_cell = heights
        self.cols = []
        self.rows = []
        self.cells = [[] for _ in range(len(heights))]
        for i, w in enumerate(widths[:-1]):
            self.cols.append(Col(width=w, margin_left=margin_left, margin_right=margin_right,
                                 index=i, line_left=Line(2, 1), line_right=Line(0, 0)))
        self.cols.append(Col(width=widths[-1], margin_left=margin_left, margin_right=margin_right,
                             index=len(widths), line_left=Line(2, 1), line_right=Line(2, 1)))

        for i, h in enumerate(heights[:-1]):
            self.rows.append(Row(height=h, margin_top=margin_top, margin_bottom=margin_bottom,
                                 index=i, line_top=Line(2, 1), line_bottom=Line(0, 0)))
        self.rows.append(Row(height=heights[-1], margin_top=margin_top, margin_bottom=margin_bottom,
                             index=len(heights), line_top=Line(2, 1), line_bottom=Line(2, 1)))

        self.table_height = sum([r.get_height() for r in self.rows])
        self.table_width = sum([c.get_width() for c in self.cols])

        for i, c in enumerate(self.cols):
            for r in self.rows:
                self.cells[i].append(Cell(c, r, EMPTY, i))

    def get_col_start(self, index):
        return sum([c.get_width() for c in self.cols[:index]])

    def get_row_start(self, index):
        return sum([r.get_height() for r in self.rows[:index]])

    def draw(self):
        fonts = ['font/times.ttf', 'font/timesbd.ttf', 'font/timesi.ttf', 'font/timesbi.ttf']
        img = np.ones((self.table_height, self.table_width, 3), dtype=np.uint8) + 244
        for idx, row in enumerate(self.rows):
            start = self.get_row_start(idx)
            # line top
            img = self.draw_line(line=row.line_top, img=img, xmin=start, xmax=start + row.line_top.line_thichness)
            img[start: start + row.line_top.line_thichness, :, ] = 0
            # # margin top
            # img[start + row.line_top.line_thichness:
            #     start + row.line_top.line_thichness + row.margin_top, :, ] = (255, 0, 0)
            # # margin bot
            # img[start + row.get_height() - row.margin_bottom - row.line_bottom.line_thichness:
            #     start + row.get_height() - row.line_bottom.line_thichness, :, ] = (0, 255, 0)
            # line bot
            img[start + row.get_height() - row.line_bottom.line_thichness:
                start + row.get_height(), :, ] = 0

        for idx, col in enumerate(self.cols):
            start = self.get_col_start(idx)
            # line left
            img[:, start:start + col.line_left.line_thichness, :] = 0
            # # margin left
            # img[:, start + col.line_left.line_thichness:
            #        start + col.line_left.line_thichness + col.margin_left, :] = (255, 0, 0)
            # # margin right
            # img[:, start + col.get_width() - col.margin_right - col.line_right.line_thichness:
            #        start + col.get_width() - col.line_right.line_thichness, :] = (0, 255, 0)
            # line right
            img[:, start + col.get_width() - col.line_right.line_thichness:
                   start + col.get_width()] = 0
        text = [['', '30/06/2019', '30/06/2018'],
                ['', '', ''],
                ['Công ty CP Chế Biến Thực Phẩm Xuất Khẩu Kiên Giang', '500.000.000', '500.000.000'],
                ['Công ty CP XD GT Thủy lợi Kiên Giang', '450.000.000', '450.000.000'],
                ['Công ty CP Bắc Trung Bộ', '190.000.000', '190.000.000'],
                ['Công ty CP Nhựa Trường Thịnh', '2.000.000.000', '2.000.000.000'],
                ['', '', ''],
                ['Cộng', '3.140.000.000', '3.140.000.000']]
        for i_r, row in enumerate(self.rows):
            for i_c, col in enumerate(self.cols):
                xmin, ymin, xmax, ymax = self.get_bounding_box_by_xy(i_r, i_c)
                img = draw_text(text=text[i_r][i_c], font=fonts[0], align='center', size=36, xmin=ymin, ymin=xmin,
                                xmax=xmax, ymax=ymax, img=img, text_color='#000000,#282828')
        show_img(img)

    def draw_line(self, line: Line, img, xmin, xmax, ymin, ymax, orient='vertical'):
        assert orient in ['vertical', 'horizontal']
        if line.type == 0:
            return
        line_thichness = line.line_thichness // (2 * line.type - 1)
        for i in range(2 * line.type - 1):
            if i % 2 == 0:
                if orient == 'horizontal':
                    start = xmin + i * line_thichness
                    img[start:start + line_thichness, ymin:ymax, :] = 0
                else:
                    start = ymin + i * line_thichness
                    img[xmin:xmax, start:start + line_thichness, :] = 0
        return img

    def get_bounding_box_by_xy(self, x, y):
        # row
        xmin = self.get_row_start(x) + self.rows[x].line_top.line_thichness + self.rows[x].margin_top
        xmax = self.get_row_start(x) + self.rows[x].get_height() - self.rows[x].line_bottom.line_thichness
        # col
        ymin = self.get_col_start(y) + self.cols[y].line_left.line_thichness + self.cols[y].margin_left
        ymax = self.get_col_start(y) + self.cols[y].get_width() - self.cols[y].line_right.line_thichness
        return xmin, ymin, xmax, ymax


def draw_text(text, font, align, size, xmin, ymin, xmax, ymax, img, text_color='#000000,#282828'):
    assert align in ['left', 'right', 'center', 'jutify']
    if type(img) == np.ndarray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)  # BGR

    image_font = ImageFont.truetype(font=font, size=size)
    words = text.split(" ")
    space_width = image_font.getsize(" ")[0] * 1
    space_height = image_font.getsize(" ")[1] * 1
    words_width = [image_font.getsize(w)[0] for w in words]
    text_width = sum(words_width) + int(space_width) * (len(words) - 1)
    text_height = max([image_font.getsize(w)[1] for w in words])

    txt_draw = ImageDraw.Draw(img)
    colors = [ImageColor.getrgb(c) for c in text_color.split(",")]
    c1, c2 = colors[0], colors[-1]

    fill = (
        random.randint(min(c1[0], c2[0]), max(c1[0], c2[0])),
        random.randint(min(c1[1], c2[1]), max(c1[1], c2[1])),
        random.randint(min(c1[2], c2[2]), max(c1[2], c2[2])),
    )
    for i, w in enumerate(words):
        start_x = sum(words_width[0:i]) + i * int(space_width) + xmin
        start_y = 0 + ymin
        txt_draw.text(
            (start_x, start_y),
            w,
            fill=fill,
            font=image_font,
        )
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


Table().draw()
# img = np.zeros((500, 500, 3), dtype=np.uint8) + 255
# fonts = ['font/times.ttf', 'font/timesbd.ttf', 'font/timesi.ttf', 'font/timesbi.ttf']
# draw_text(text="Đào Ngọc An", font=fonts[0], align='center', size=48, xmin=200, ymin=200, xmax=500, ymax=500, img=img,
#           text_color='#000000,#282828').show()

# chữ viết thường, chữ nghiêng, chữ viết đậm, chữ viết đậm nghiêng
#
