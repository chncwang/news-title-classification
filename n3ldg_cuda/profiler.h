#ifndef N3LDG_CUDA_PROFILER_H
#define N3LDG_CUDA_PROFILER_H

#include <string>
#include <map>
#include <chrono>
#include <utility>
#include <iostream>
#include <stack>


namespace n3ldg_cuda {

struct Event {
    std::string name;
    int count;
    float total_time_in_nanoseconds;

    Event(std::string &&name, int count, float total_time_in_nanoseconds) {
        this->name = std::move(name);
        this->count = count;
        this->total_time_in_nanoseconds = total_time_in_nanoseconds;
    }

    Event() = default;
    Event(const Event &event) = default;
};

struct Elapsed {
    typedef
        std::chrono::time_point<std::chrono::high_resolution_clock> Timestamp;
    Timestamp begin;
    Timestamp end;
    std::string name;
};

class Profiler {
public:
    static Profiler &Ins() {
        static Profiler *p;
        if (p == NULL) {
            p = new Profiler;
        }
        return *p;
    }

    void BeginEvent(const std::string &name) {
        Elapsed elapsed;
        elapsed.name = name;
        running_events_.push(std::move(elapsed));
        running_events_.top().begin = std::chrono::high_resolution_clock::now();
    }

    void EndEvent() {
        Elapsed &top = running_events_.top();
        top.end = std::chrono::high_resolution_clock::now();
        std::string name = top.name;
        float time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                top.end - top.begin).count();
        auto it = event_map_.find(top.name);
        if (it == event_map_.end()) {
            Event event(std::move(top.name), 1, time);
            event_map_.insert(std::make_pair(event.name, event));
        } else {
            Event &event = it->second;
            event.count++;
            event.total_time_in_nanoseconds += time;
        }
        if (running_events_.size() == 1) {
            root = &event_map_.at(name);
        }
        running_events_.pop();
    }

    void EndCudaEvent();

    void Print() {
        assert(running_events_.empty());
        std::vector<Event> events;
        for (auto &it : event_map_) {
            Event &event = it.second;
            events.push_back(event);
        }

        std::sort(events.begin(), events.end(), [](const Event &a,
                    const Event &b) {return a.total_time_in_nanoseconds >
                b.total_time_in_nanoseconds;});

        for (Event &event : events) {
            std::cout << "name:" << event.name << " count:" << event.count <<
                " total time:" << event.total_time_in_nanoseconds / 1000000.0
                << "avg:" << event.total_time_in_nanoseconds / event.count / 1000000 <<
                " ratio:" << event.total_time_in_nanoseconds /
                root->total_time_in_nanoseconds << std::endl;
        }
    }

private:
    Profiler() = default;
    std::map<std::string, Event> event_map_;
    std::stack<Elapsed> running_events_;
    Event *root = NULL;
};

}

#endif
