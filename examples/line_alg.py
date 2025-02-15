def _drawLineH(x0, y0, x1, y1, image, color):
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    
    dx = x1 - x0
    dy = y1 - y0

    dir = -1 if dy < 0 else 1
    dy *= dir
    
    y = y0
    p = 2*dy - dx
    for i in range(dx + 1):
        _drawPixel(x0 + i, y, image, color)
        if p >= 0:
            y += dir
            p -= 2*dx
        p += 2*dy

def _drawLineV(x0, y0, x1, y1, image, color):
    if y0 > y1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    
    dx = x1 - x0
    dy = y1 - y0

    dir = -1 if dx < 0 else 1
    dx *= dir
    
    x = x0
    p = 2*dx - dy
    for i in range(dy + 1):
        _drawPixel(x, y0 + i, image, color)
        if p >= 0:
            x += dir
            p -= 2*dy
        p += 2*dx

def drawLine(x0, y0, x1, y1, image, color):
    if abs(y1 - y0) < abs(x1 - x0):
        _drawLineH(x0, y0, x1, y1, image, color)
    else:
        _drawLineV(x0, y0, x1, y1, image, color)

def _drawPixel(x, y, image, color):
    image[y, x] = color

if __name__ == '__main__':
    pass