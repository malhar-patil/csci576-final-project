#pragma once

#include<QWidget>
#include<QMouseEvent>
#include<QPaintEvent>
#include<vector>
#include"SegmentData.h"

class TimelineBar : public QWidget{
    Q_OBJECT
    
    public:
    TimelineBar(QWidget* parent = nullptr);

    signals:
    void seekRequested(double sec);

    public slots:
    void setSegments(const std::vector<Segment>& segments, double totalDuration);
    void setPlayhead(double sec);

    protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;

    private:
    double m_totalDuration = 0.0;
    std::vector<Segment> m_segments;
    double m_playhead = 0.0;
};

