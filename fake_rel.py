import os
import random

import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageColor


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

    def __init__(self, col: Col, row: Row, text: str, font: str, align: str, size: int, cell_id: int):
        self.col = col
        self.row = row
        self.text = text
        self.cell_id = cell_id
        self.font = font
        self.align = align
        self.size = size


class Table(object):
    def __init__(self, widths: list = [0.6, 0.2, 0.2], table_widths=1500, table_height=500,
                 margin_left=10, margin_right=10, margin_top=5, margin_bottom=1):

        self.n_rows = table_height // Row(height=int(41 * table_widths / 1500), margin_top=margin_top,
                                          margin_bottom=margin_bottom,
                                          index=-1, line_top=Line(2, 1), line_bottom=Line(6, 2)).get_height()
        self.n_cols = len(widths)
        self.heigh_each_cell = [50] * self.n_rows
        self.width_each_cell = list(map(lambda x: int(table_widths * x), widths))

        self.cols = []
        self.rows = []
        self.cells = [[] for _ in range(self.n_rows)]

        #
        for i, w in enumerate(self.width_each_cell[:-1]):
            self.cols.append(Col(width=w, margin_left=margin_left, margin_right=margin_right,
                                 index=i, line_left=Line(2, 1), line_right=Line(0, 0)))
        self.cols.append(Col(width=self.width_each_cell[-1], margin_left=margin_left, margin_right=margin_right,
                             index=len(self.width_each_cell), line_left=Line(2, 1), line_right=Line(2, 1)))

        for i, h in enumerate(self.heigh_each_cell[:-1]):
            self.rows.append(Row(height=h, margin_top=margin_top, margin_bottom=margin_bottom,
                                 index=i, line_top=Line(2, 1), line_bottom=Line(0, 0)))
        self.rows.append(Row(height=self.heigh_each_cell[-1], margin_top=margin_top, margin_bottom=margin_bottom,
                             index=len(self.heigh_each_cell), line_top=Line(2, 1), line_bottom=Line(6, 2)))

        self.table_height = sum([r.get_height() for r in self.rows])
        self.table_width = sum([c.get_width() for c in self.cols])

    def get_col_start(self, index):
        return sum([c.get_width() for c in self.cols[:index]])

    def get_row_start(self, index):
        return sum([r.get_height() for r in self.rows[:index]])

    def get_table_bounding_box(self):
        return (self.table_height, self.table_width)

    def draw(self, background_color=255):
        img = np.zeros((self.table_height, self.table_width, 3), dtype=np.uint8)
        img[:, :] = background_color
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                img = self.show_cell_by_xy(img, i, j, left=True, right=True, top=True, bottom=True,
                                           margin_left=False, margin_right=False)
        return img


    def draw_line(self, line: Line, img, xmin, xmax, ymin, ymax, orient='vertical'):
        assert orient in ['vertical', 'horizontal']
        if line.type == 0:
            return img
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

    def show_cell_by_xy(self, img, x: int, y: int, left=False, right=False, top=False, bottom=False,
                        margin_left=False, margin_right=False, margin_top=False, margin_bottom=False):
        xmin = self.get_row_start(x) if not margin_top else self.get_row_start(x) + self.rows[x].margin_top
        xmax = xmin + self.rows[x].get_height() if not margin_bottom else xmin + self.rows[x].get_height() \
                                                                          - self.rows[x].margin_bottom
        ymin = self.get_col_start(y) if not margin_left else self.get_col_start(y) + self.cols[y].margin_left
        ymax = ymin + self.cols[y].get_width() if not margin_right else self.get_col_start(y) + self.cols[y].get_width() \
                                                                        - self.cols[y].margin_right

        if left:
            img = self.draw_line(line=self.cols[y].line_left, img=img, orient="vertical",
                                 xmin=xmin,
                                 xmax=xmax,
                                 ymin=ymin,
                                 ymax=ymin + self.cols[y].line_left.line_thichness)
        if right:
            img = self.draw_line(line=self.cols[y].line_right, img=img, orient="vertical",
                                 xmin=xmin,
                                 xmax=xmax,
                                 ymin=ymax - self.cols[y].line_right.line_thichness,
                                 ymax=ymax)
        if top:
            img = self.draw_line(line=self.rows[x].line_top, img=img, orient="horizontal",
                                 xmin=xmin,
                                 xmax=xmin + self.rows[x].line_top.line_thichness,
                                 ymin=ymin,
                                 ymax=ymax)
        if bottom:
            img = self.draw_line(line=self.rows[x].line_bottom, img=img, orient="horizontal",
                                 xmin=xmax - self.rows[x].line_bottom.line_thichness,
                                 xmax=xmax,
                                 ymin=ymin,
                                 ymax=ymax)
        return img

    def get_bounding_box_by_xy(self, x, y):
        # row
        xmin = self.get_row_start(x) + self.rows[x].line_top.line_thichness + self.rows[x].margin_top
        xmax = self.get_row_start(x) + self.rows[x].get_height() \
               - self.rows[x].line_bottom.line_thichness - self.rows[x].margin_bottom
        # col
        ymin = self.get_col_start(y) + self.cols[y].line_left.line_thichness + self.cols[y].margin_left
        ymax = self.get_col_start(y) + self.cols[y].get_width() \
               - self.cols[y].line_right.line_thichness - self.cols[y].margin_right
        return xmin, ymin, xmax, ymax


if __name__ == '__main__':
    show_img(Table().draw(255))
# img = np.zeros((500, 500, 3), dtype=np.uint8) + 255
# fonts = ['font/times.ttf', 'font/timesbd.ttf', 'font/timesi.ttf', 'font/timesbi.ttf']
# draw_text(text="Đào Ngọc An", font=fonts[0], align='center', size=48, xmin=200, ymin=200, xmax=500, ymax=500, img=img,
#           text_color='#000000,#282828').show()

# chữ viết thường, chữ nghiêng, chữ viết đậm, chữ viết đậm nghiêng
#
