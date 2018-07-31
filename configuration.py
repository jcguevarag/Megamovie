#class Configuration:
#    def __init__(self, smallest_row, smallest_col, greatest_row, greatest_col, superimposed_image, individual_image, angle):
#        self.smallest_row = smallest_row
#        self.smallest_col = smallest_col
#        self.greatest_row = greatest_row
#        self.greatest_col = greatest_col
#        self.superimposed_image = superimposed_image
#        self.individual_image = individual_image
#        self.angle = angle
#        self.area = (greatest_row[0] - smallest_row[0])*(greatest_col[1] - smallest_col[1])

#    def __cmp__(self, other):
#        return self.area.__cmp__(other.area)

#    def __lt__(self, other):
#        return self.area < other.area

class Configuration:
    def __init__(self, area, superimposed_image, individual_image, angle):
        self.superimposed_image = superimposed_image
        self.individual_image = individual_image
        self.angle = angle
        self.area = area

    def __cmp__(self, other):
        return self.area.__cmp__(other.area)

    def __lt__(self, other):
        return  self.area < other.area

    def __str__(self):
        return '{}, {}'.format(self.area, self.angle)