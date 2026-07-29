#include <QApplication>
#include <QMainWindow>
#include <QMediaPlayer>
#include <QVideoWidget>
#include <QPushButton>
#include <QSlider>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QDir>
#include <QAudioOutput>
#include <QTimer>
#include <vector>
#include <QListWidget>
#include "SegLoader.h"
#include "TimelineBar.h"
 

int main(int argc, char* argv[]){
    std::vector<Segment> segments;

    QApplication app(argc, argv);
    
    QMainWindow window;



    
    QWidget* central = new QWidget;
    QHBoxLayout* rootLayout = new QHBoxLayout(central);

    QWidget* leftPanel = new QWidget;
    QVBoxLayout* layout = new QVBoxLayout(leftPanel);

    QListWidget* sidebar = new QListWidget;
    sidebar->setFixedWidth(200);

    rootLayout->addWidget(leftPanel,1);
    rootLayout->addWidget(sidebar);


    QVideoWidget* videoWidget = new QVideoWidget();
    layout->addWidget(videoWidget, 1);

    TimelineBar* timelineBar = new TimelineBar;
    timelineBar->setFixedHeight(40);
    layout->addWidget(timelineBar);

    QMediaPlayer* player = new QMediaPlayer();
    QAudioOutput* audio = new QAudioOutput();



    player->setVideoOutput(videoWidget);
    player->setAudioOutput(audio);
    QObject::connect(player, &QMediaPlayer::positionChanged, [&](qint64 sec){
        timelineBar->setPlayhead((double) sec/1000);
    });

    QObject::connect(timelineBar, &TimelineBar::seekRequested, [&](double sec){
        player->setPosition((qint64) sec*1000);
    });

    QTimer* skipTimer = new QTimer();
    skipTimer->setInterval(200);
    QObject::connect(skipTimer, &QTimer::timeout, [&](){
        if(segments.empty()){
            return;
        }
        double currPosition = player->position() / 1000;
        for(auto& segment : segments){
            if(!segment.enabled && currPosition >= segment.start_sec && currPosition < segment.end_sec){
                player->setPosition((qint64) (segment.end_sec*1000)+1000);
                player->play();
                break;
            }
        }

    });
    skipTimer->start();



    //player->setSource(QUrl::fromLocalFile("videos_with_ads/test_001.mp4"));

    //! Controls panel
    auto formatTime = [](qint64 value) -> QString{
        int sec = value/1000;
        return QString("%1:%2:%3")
        .arg(sec/3600)
        .arg(sec%3600 / 60, 2, 10, QChar('0'))
        .arg(sec%60,2,10,QChar('0'));
    };

    //? play-pause button
    QPushButton* playPauseButton = new QPushButton("Play");
    QObject::connect(playPauseButton, &QPushButton::clicked, [&]() {
        if(player->playbackState() == QMediaPlayer::PlayingState){
            playPauseButton->setText("Play");
            player->pause();
        }
        else{
            playPauseButton->setText("Pause");
            player->play();
        }
    });
    

    //? openFileButton
    QPushButton* openFileButton = new QPushButton("Open File");
    QObject::connect(openFileButton, &QPushButton::clicked, [&](){
        QString filePath = QFileDialog::getOpenFileName(
            nullptr,
            "Open Video",
            QDir::homePath(),
            "Video Files (*.mp4 *.mkv *.avi)",
            nullptr,
            QFileDialog::DontUseNativeDialog
        );
        if (!filePath.isEmpty()){
            player->setSource(QUrl::fromLocalFile(filePath)); 
            player->play();
            segments = SegLoader::load(filePath);
            if(!segments.empty()){
                timelineBar->setSegments(segments, segments.back().end_sec);
            }
            else{
                timelineBar->setSegments(segments, 0.0);
            }
            playPauseButton->setText("Pause"); 

            sidebar->clear();
            for(auto& segment : segments){
                if(segment.type == SegmentType::non_content){
                    sidebar->addItem("Non-content " + formatTime((qint64) segment.start_sec * 1000));
                }
                else{
                    sidebar->addItem("Content "+formatTime((qint64) segment.start_sec * 1000));
                }
                
            }

        }
    });

    bool skipNonContent = false;
    QPushButton* skipAdsButton = new QPushButton("Skip non-content");
    QObject::connect(skipAdsButton, &QPushButton::clicked, [&](){
        for(auto& segment : segments){
            if(segment.type == SegmentType::non_content){
                if(segment.enabled){
                    segment.enabled = false;
                }
                else{
                    segment.enabled = true;
                }
            }
        }
        if(!skipNonContent){
            skipNonContent = true;
            skipAdsButton->setText("Un-skip non-content");
        }
        else{
            skipNonContent = false;
            skipAdsButton->setText("Skip non-content");  
        }
    });

    //? timelabel
    QLabel* timeLabel = new QLabel();
    timeLabel->setText("0:00/0:00");

    //? volume slider
    QSlider* volumeSlider = new QSlider(Qt::Horizontal);
    volumeSlider->setValue(70);
    audio->setVolume(0.7);
    volumeSlider->setRange(0,100);
    //volumeSlider->setFixedWidth();
    QObject::connect(volumeSlider, &QSlider::valueChanged, [&](int value){
        audio->setVolume((float)value/100);
    });

    //! video seek slider
    QSlider* seekSlider = new QSlider(Qt::Horizontal);
    seekSlider->setRange(0,0);
    QObject::connect(player, &QMediaPlayer::durationChanged, [&](qint64 duration){
        seekSlider->setRange(0,duration);
    });
    
    QObject::connect(player, &QMediaPlayer::positionChanged, [&](qint64 position){
        if(!seekSlider->isSliderDown()){
            seekSlider->setValue(position);
            timeLabel->setText(formatTime(position) + " / " + formatTime(player->duration()));

            double currSec = position/1000.0;
            for(int i=0; i<segments.size(); i++){
                if(currSec >= segments[i].start_sec && currSec < segments[i].end_sec){
                    sidebar->setCurrentRow(i);
                    break;
                }
            }  
        }
    });

    QObject::connect(seekSlider, &QSlider::sliderMoved, [&](int value){
        player->setPosition(value);
    });

    QObject::connect(seekSlider, &QSlider::sliderReleased, [&](){
        player->setPosition(seekSlider->value());
    });
    //
    QHBoxLayout* controlsRow = new QHBoxLayout();
    layout->addWidget(seekSlider);
    layout->addLayout(controlsRow);
    controlsRow->addWidget(openFileButton);
    controlsRow->addWidget(playPauseButton);
    controlsRow->addWidget(skipAdsButton);
    controlsRow->addStretch();
    controlsRow->addWidget(timeLabel);
    controlsRow->addStretch();
    controlsRow->addWidget(volumeSlider);
    

    window.setCentralWidget(central);
    window.resize(1024,640);
    window.show();
    return app.exec();
}