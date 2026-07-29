#include "TimelineBar.h"
#include<QPainter>


TimelineBar :: TimelineBar(QWidget* parent) : QWidget(parent){

}

void TimelineBar :: setSegments(const std::vector<Segment>& segments, double totalDuration){
    m_segments = segments;
    m_totalDuration = totalDuration;
    update();
}

void TimelineBar :: setPlayhead(double sec){
    m_playhead = sec;
    update();
}

void TimelineBar :: paintEvent(QPaintEvent* event){
    if(m_totalDuration == 0.0 || m_segments.empty()){
        return;
    }

    QPainter painter(this);
    painter.fillRect(rect(), QColor("#000000"));

    for(const Segment& segment : m_segments){
        double x = (segment.start_sec / m_totalDuration) * width();
        double w = ((segment.end_sec - segment.start_sec) / m_totalDuration) * width();
        QColor blockColor;
        if(segment.type == SegmentType::content){
            blockColor = ("#00a500");
        }
        else{
            blockColor = ("#CC3333");
        }
        painter.fillRect(x, 0 , w, height(),  blockColor);
    }
    double playheadX = (m_playhead / m_totalDuration) * width();
    painter.setPen(Qt::white);
    painter.drawLine(playheadX, 0, playheadX, height());

}
void TimelineBar :: mousePressEvent(QMouseEvent* event){
    if(m_totalDuration == 0.0){
        return;
    }
    double xPos = event->pos().x();
    double sec = (xPos / width()) * m_totalDuration;
    emit seekRequested(sec);
}

